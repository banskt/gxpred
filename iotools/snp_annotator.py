
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
import math

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


# def myExpcurve(xvals, w, lambd):
#     yvalues = 1 - w + w*( lambd*np.exp(-lambd*(np.abs(xvals))) )
#     return yvalues

# def myExpcurveMix(xvals, lambd1, lambd2, t1, t2):
#     yvalues = lambd2*np.exp(-(1/t2)*(np.abs(xvals))) +  lambd1*np.exp(-(1/t1)*(np.abs(xvals)))
#     return yvalues

def myTcurve(xvals):
	v =  1.193099602813465
	d0 = 313651.3878703753
	scale = 7.880944428318826e-07
	w = 0.7282588550247171
	x = xvals / d0
	tvalues = (math.gamma((v+1)/2)/(math.sqrt(v*np.pi)*math.gamma(v/2)) ) * (1+ (x**2)/v)**(-(v+1)/2) 
	return tvalues

def get_DHS_feature(snps, gene_info):
	start = gene_info.start
	end = gene_info.end
	dist_arr = []
	for snp in snps:
		dist = (start - snp.bp_pos)
		dfeat = myTcurve(dist)
		dist_arr.append(1/dfeat)
	return np.array(dist_arr)

def get_distance_feature(selected_snps, gene, usedist):
	nsnps_used = len(selected_snps)
	if usedist == "dhs":
		print("Running with distance feature")
		dist_feature = get_DHS_feature(selected_snps, gene)
	if usedist == "random":
		print("Running with randomized distance feature")
		dist_feature = get_DHS_feature(selected_snps, gene)
		np.random.shuffle(dist_feature)
	if usedist == "nodist":
		dist_feature = np.ones(nsnps_used)
	return dist_feature

def get_features(selected_snps, usefeat):
	# TO-DO:
	# - Add histone marks
	# - Add gencode annotations
	# first feature must always be a row of 1's
	nsnps_used = len(selected_snps)
	feature1 = np.ones((nsnps_used, 1))
	if feat == "nofeat":
		features = feature1    
	if feat == "randomint":
		feature2 = np.random.random_integers(0,1, nsnps_used)[np.newaxis].T
		features = np.concatenate((feature1,feature2), axis=1)
	return features

def get_promoter_annotation(gene, selected_snps):
	start = gene_info.start
	promoter_arr = np.zeros((len(selected_snps),0))
	for i, snp in enumerate(snps):
		dist = (start - snp.bp_pos)
		if dist <= 2000:
			promoter_arr[i] = 1
	return promoter_arr

def get_GENCODE_annotation(gencode_file, gene, selected_snps, feat_type):
	gencode_data = get_GENCODE_data(gencode_file, gene, feat_type)
	return get_feature(gencode_data, selected_snps)

def get_GENCODE_data(gencode_file, gene, feat_type):
	from utils.containers import GeneInfo
	import gzip

	if feat_type not in ["UTR", "exon"]:
		raise Exception("No such feature type "+feat_type)

	w_start = gene.start - 1000000
	w_end = gene.end + 1000000

	trim = False
	annotfile = gencode_file
	geneinfo = list()
	try:
		with gzip.open(annotfile, 'r') as mfile:
			for line in mfile:
				linesplit = line.decode().strip().split('\t')

				if linesplit[0][0] == '#' or linesplit[2] == 'gene': continue # skip header
				chrom = linesplit[0][3:]
				if chrom != "{:d}".format(gene.chrom): continue

				infolist = linesplit[8].split(';')
				rowtype = infolist[2].strip().split(' ')[1].replace('"','')

				#         explore tags?
				#         for e in infolist:
				#             arr = e.strip().split(' ')
				#             if arr[0] == "tag":
				
				gene_id = infolist[0].strip().split(' ')[1].replace('"','')
				if trim:
					gene_id = gene_id.split(".")[0]
				# if gene.ensembl_id != gene_id: continue
				if linesplit[2] != feat_type: continue

				# TSS: gene start (0-based coordinates for BED)
				if linesplit[6] == '+':
					start = np.int64(linesplit[3]) - 1
					end   = np.int64(linesplit[4])
				elif linesplit[6] == '-':
					start = np.int64(linesplit[3])  # last base of gene
					end   = np.int64(linesplit[4]) - 1
				else:
					raise ValueError('Strand not specified.')

				if start >= w_start and end <= w_end:
					gene_name = infolist[4].strip().split(' ')[1].replace('"','')
					this_gene = GeneInfo(name       = "UTR",
										 ensembl_id = gene_id,
										 chrom      = chrom,
										 start      = start,
										 end        = end)
				else:
					continue

				geneinfo.append(this_gene)
	except IOError as err:
		raise IOError('{:s}: {:s}'.format(annotfile, err.strerror))

	return geneinfo

def get_feature(gencode_data, selected_snps):
	feature = np.zeros((len(selected_snps), 1))
	for i, snp in enumerate(selected_snps):
	    for utr in gencode_data:
	        if snp.bp_pos >= utr.start and snp.bp_pos <= utr.end:
	            feature[i] = 1
	            break
	return feature