
# A general rule of thumb is that:

# active promoter: H3K4me3, H3K9Ac
# active enhancer: H3K4me1, H3K27Ac
# active elongation: H3K36me3, H3K79me2
# repressed promoters and broad regions: H3K27me3, H3K9me3

# H3K4me3
# H3K4me1
# H3K27Ac
# H3K9Ac
# H3K36me3
# H3K79me2

import numpy as np

def get_dummy_dist_feature(snps, gene_info, window):
	start = gene_info.start
	end = gene_info.end
	dist_arr = []
	for snp in snps:
		if snp.bp_pos <= start:
			dist = 1 - (start - snp.bp_pos) / window
		elif snp.bp_pos > start:
			dist = 1 - (snp.bp_pos - start) / window
		dist_arr.append(dist)
	return np.array(dist_arr)[np.newaxis].T