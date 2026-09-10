# Architecture

Input images enter Mode G through metadata, lunar CRS, footprint/catalog overlap, metadata prior, bidirectional matcher, cycle and uncertainty filters, residual geometry, constrained refinement, and quality gate. Mode V performs DINO/FAISS candidate retrieval over the local catalog before the same verification path. The matcher never has final authority.
