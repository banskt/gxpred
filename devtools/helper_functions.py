

import numpy as np
import os
from scipy.stats import pearsonr

def write_params(outdir, params, overwrite=True):
    if os.path.exists(outdir):
        if not overwrite:
            with open("error.log", 'a') as outstream:
                outstream(params[0]+" - Folder with previous results exists! check!\n")
            raise Exception("Folder with previous results exists! check!")
    else:
        os.makedirs(outdir)
    with open(os.path.join(outdir, "params.txt"), 'w') as outstream:
        headers = ["Prior","pi","mu","sigma","sigmabg","tau","pi_prior","mu_prior","sigma_prior","sigmabg_prior","tau_prior"]
        dict_values = []
        if params[3] != None:
            for i in params[3].keys():
                headers.append(i)
                dict_values.append(str(params[3][i]))
        data = [params[0]] + [str(item) for sublist in params[1:3] for item in sublist] + dict_values
        outdict = dict(zip(headers, data))
        for i in outdict:
            outstream.write(i+"\t"+outdict[i]+"\n")


def load_target_genes(genelistfile, gene_info, chrom=None, chroms=range(1,23)):
    gene_list = []
    r2val = []
    predixcan_r2val = []
    with open(genelistfile, 'r') as instream:
        for line in instream:
            arr = line.strip().split()
            gene_list.append(arr[0])
            r2val.append(arr[1])
            predixcan_r2val.append(arr[2])

    print("Read {:d} genes with high r2 values\n".format(len(gene_list)))
            
    if chrom:
        chroms = [chrom]
    selected_gene_ids = []
    for i in gene_info:
        trimid = i.ensembl_id.split(".")[0]
        if trimid in gene_list and i.chrom in chroms:
            # print("Gene {:s}, CHR {:d}, R2 value: {:s} ".format(i.ensembl_id, i.chrom, r2val[gene_list.index(trimid)]))
            selected_gene_ids.append(i.ensembl_id)
    print("Found {:d} genes in CHR {:s}".format(len(selected_gene_ids), ",".join(list(map(str,chroms)))))
    return selected_gene_ids

def write_r2_dataframe(modelpath, chrom, prior, r_values, prediction_obj, overwrite=False):
    import pandas as pd

    outtable = os.path.join(modelpath, "genes_r2.txt")
    if not os.path.exists(outtable) or overwrite:
        # Read genes.txt table with learned parameters and fix some columns and row namings
        learning_table = os.path.join(modelpath, "chr{:d}".format(chrom), "genes.txt")
        genes_df = pd.read_table(learning_table, header = 0, sep='\s+')
        newcolumns = list(genes_df.columns)
        newcolumns = [i.strip() for i in newcolumns]
        genes_df.columns = newcolumns
        genes_df["Ensembl_ID"] = [str(i).split(".")[0] for i in genes_df["Ensembl_ID"]]
    else:
        genes_df = pd.read_table(outtable, header=0)
        
    genes_df.index = [str(i).split(".")[0] for i in genes_df["Ensembl_ID"]]

    new_df = pd.DataFrame(list(r_values**2), columns=[prior])
    new_df.index = prediction_obj.sorted_gene_names
    new_df["Ensembl_ID"] = [str(i).split(".")[0] for i in prediction_obj.sorted_gene_names]

    compiled_table = pd.merge(genes_df, new_df, how='outer', on=genes_df.columns[0])
    compiled_table.to_csv(outtable,sep="\t", float_format='%.5f', index=False, na_rep="NA")

def write_predicted_r2(outfile, prior, params, gxpred_r, predixcan_r, gene_names):
    mode = "w"
    if os.path.exists(outfile):
        mode = "a"
        with open(outfile,'r') as instream:
            prev_genes = instream.readline().split()[6:]
            common_genes, ix = get_common_elements(gene_names, prev_genes)

    if len(gxpred_r) == len(gene_names) and len(predixcan_r) == len(gene_names):
        with open(outfile, mode) as outstream:
            if mode == "w":
                headers = "\t".join(["prior", "pi", "mu", "sigma", "sigmabg", "tau"]+ gene_names)
                outstream.write(headers+"\n")
                # write predixcan values for control
                values = list(map("{:.3f}".format,params + list(predixcan_r**2)))
                predline = "\t".join(["predixcan"]+ values)
                outstream.write(predline+"\n")
            else:
                with open(outfile, 'r') as instream:
                    if not gene_names == instream.readline().split()[6:]:
                        raise Exception("Gene names mismatch")
                values = list(map("{:.3f}".format,params + list(gxpred_r**2)))
                predline = "\t".join([prior]+ values)
                outstream.write(predline+"\n")
    else:
        raise Exception("Different number of genes found")

def new_write_predicted_r2(outfile, prior, params, gxpred_r, predixcan_r, gene_names):
    import pandas as pd
    # if os.path.exists(outfile):
    if False:
        pred_df = pd.read_table(outfile, header=0)
        new_df = pd.DataFrame([[prior] + params + list(gxpred_r**2)], columns= ["prior", "pi", "mu", "sigma", "sigmabg", "tau"]+ gene_names )
        result_df = pd.concat([pred_df, new_df], axis=1)
        result_df.to_csv(outfile, sep="\t")
    else:
        if len(gxpred_r) == len(gene_names) and len(predixcan_r) == len(gene_names):
            with open(outfile+"."+prior, 'w') as outstream:
                headers = "\t".join(["prior", "pi", "mu", "sigma", "sigmabg", "tau"]+ gene_names)
                outstream.write(headers+"\n")
                # write predixcan values for control
                values = list(map("{:.3f}".format,params + list(predixcan_r**2)))
                predline = "\t".join(["predixcan"]+ values)
                outstream.write(predline+"\n")

                # write gxpred values
                values = list(map("{:.3f}".format,params + list(gxpred_r**2)))
                predline = "\t".join([prior]+ values)
                outstream.write(predline+"\n")


def pearson_corr_rowwise(mat1, mat2):
    results = list()
    good = 0
    bad = 0
    for i,predrow in enumerate(mat1):   # one gene at the time
        exprow = mat2[i,:]
        if np.all(predrow == 0):
            res = (0,1)
            bad += 1
        else:
            res = pearsonr(predrow,exprow)
            good += 1
        results.append(np.float64(res[0]))
    return np.array(results)


def get_common_elements(mylist, targetlist):
    common_elements = [g for g in targetlist if g in mylist]
    if len(common_elements) == 0:
        raise Exception("No common elements found")
    ix = [mylist.index(i) for i in common_elements]
    return common_elements, ix
