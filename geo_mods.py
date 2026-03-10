# Backup code of the modfied XarrayDataset in geo.py in the torchgeo source code since they are not tracked in git


class XarrayDataset(GeoDataset):
    """Abstract base class for :class:`GeoDataset` stored as raster files.

    .. versionadded:: 0.8
    """

    def __init__(
        self,
        paths: Path | Iterable[Path] = 'data',
        crs: CRS | None = None,
        res: float | tuple[float, float] | None = None,
        data_vars: Sequence[str] | None = None,
        transforms: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        """Initialize a new XarrayDataset instance.

        Args:
            paths: one or more root directories to search or files to load
            crs: :term:`coordinate reference system (CRS)` to warp to
                (defaults to the CRS of the first file found)
            res: resolution of the dataset in units of CRS
                (defaults to the resolution of the first file found)
            data_vars: list of data variables to load
                (defaults to all variables of the first file found)
            transforms: a function/transform that takes an input sample
                and returns a transformed version

        Raises:
            DatasetNotFoundError: If dataset is not found.
            DependencyNotFoundError: If rioxarray is not installed.
        """
        lazy_import('rioxarray')
        xr = lazy_import('xarray')
        self.paths = paths
        self.transforms = transforms

        # Gather information about the dataset
        filepaths = []
        datetimes = []
        geometries = []
        for filepath in self.files:
            try:
                with xr.open_dataset(filepath, decode_coords='all') as src:
                    crs = crs or src.rio.crs or CRS.from_epsg(4326)
                    res = res or src.rio.resolution()
                    data_vars = data_vars or list(src.data_vars.keys())
                    tmin = pd.Timestamp(src.time.values.min())
                    tmax = pd.Timestamp(src.time.values.max())

                    if src.rio.crs is None:
                        warnings.warn(
                            f"Unable to decode coordinates of '{filepath}', "
                            f'defaulting to {crs}. Set `crs` if this is incorrect.',
                            UserWarning,
                        )
                    src = src.rio.write_crs(crs) # [BUG FIX] Modified to make sure the crs exists for each variable
                    if src.rio.crs != crs:
                        src = src.rio.reproject(crs)

                    filepaths.append(filepath)
                    datetimes.append((tmin, tmax))
                    geometries.append(shapely.box(*src.rio.bounds()))
            except (OSError, ValueError) as e:
                print("Unable to read file:", filepath, "error:", repr(e))
                # Skip files that xarray is unable to read
                continue

        if len(filepaths) == 0:
            raise DatasetNotFoundError(self)

        if res is not None:
            if isinstance(res, int | float):
                res = (res, res)

            self._res = res

        if data_vars is not None:
            self.data_vars = data_vars

        # Create the dataset index
        data = {'filepath': filepaths}
        index = pd.IntervalIndex.from_tuples(datetimes, closed='both', name='datetime')
        self.index = GeoDataFrame(data, index=index, geometry=geometries, crs=crs)

    def __getitem__(self, query: GeoSlice) -> dict[str, Any]:
        """Retrieve input, target, and/or metadata indexed by spatiotemporal slice.

        Args:
            query: [xmin:xmax:xres, ymin:ymax:yres, tmin:tmax:tres] coordinates to index.

        Returns:
            Sample of input, target, and/or metadata at that index.

        Raises:
            IndexError: If *query* is not found in the index.
        """
        x, y, t = self._disambiguate_slice(query)
        interval = pd.Interval(t.start, t.stop, closed='both') # [BUG FIX] Modified to have the same closing as interval and ensure matching
        index = self.index.iloc[self.index.index.overlaps(interval)]
        index = index.iloc[:: t.step]
        index = index.cx[x.start : x.stop, y.start : y.stop]

        if index.empty:
            raise IndexError(
                f'query: {query} not found in index with bounds: {self.bounds}'
            )

        image = self._merge_files(index.filepath, query)
        sample: dict[str, Any] = {'crs': self.crs, 'bounds': query, 'image': image}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def _merge_files(self, filepaths: Sequence[str], query: GeoSlice) -> Tensor:
        """Load and merge one or more files.

        Args:
            filepaths: one or more files to load and merge
            query: [xmin:xmax:xres, ymin:ymax:yres, tmin:tmax:tres] coordinates to index.

        Returns:
            image at that index
        """
        xr = lazy_import('xarray')
        rioxr = lazy_import('rioxarray')
        lazy_import('rioxarray.merge')

        x, y, t = self._disambiguate_slice(query)
        bounds = (x.start, y.start, x.stop, y.stop)
        res = (x.step, y.step)

        datasets = []
        for filepath in filepaths:
            src = xr.open_dataset(filepath, decode_times=True, decode_coords='all')
            src = src[self.data_vars] # [BUG FIX] Added to limit the variables to ones chosen in data_vars to save time and complexity and avoid non-existing crs errors

            if src.rio.crs is None:
                src = src.rio.write_crs(self.crs)
            
            if src.rio.crs != self.crs or res != src.rio.resolution():
                src = src.rio.reproject(self.crs, resolution=res)
            
            datasets.append(src)

        dataset = rioxr.merge.merge_datasets(
            datasets, bounds=bounds, res=res, nodata=0, crs=self.crs
        )
        dataset = dataset.sel(time=t)

        # Use array_to_tensor since merge may return uint16/uint32 arrays.
        tensors = []
        for var in self.data_vars:
            tensors.append(array_to_tensor(dataset[var].values))

        return torch.stack(tensors)