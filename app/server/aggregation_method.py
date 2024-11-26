class AggregationMethod(Enum):
    """Enumeration for Byzantine Fault Tolerant aggregation methods."""
    KRUM = "krum"
    TRIMMED_MEAN = "trimmed_mean"
    MEDIAN = "median"