# GxPRED #

This is the development branch of Gene Expression Prediction Algorithm, under the production name GxPRED.

### What is this repository for? ###

* Development of GxPRED
* v0.01

### How to run? ###
GxPRED provides core utilities / API for learning and predicting gene expression levels from genotype. Examples are shown in the base directory for learning from gEUVADIS data (`learn_from_geuvadis.py`) and predicting on GTEx data (`predict_on_gtex.py`). You can adapt these files for any dataset.

    CODEBASE = "/path/to/this/directory"
    TRAINVCF = "/path/to/gz/vcf/file/for/training"
    TRAINRPKM = "/path/to/normalized/gene/expression/file/for/training"
    TRAINGTF = "/path/to/gtf/file"
    CHROM = "21" # change it to whichever chromosome you are interested 
    PREDVCF = "/path/to/gz/vcf/for/prediction/samples"
    MODELDIR = "/path/to/directory/where/model/will/be/saved"
    
    python ${CODEBASE}/learn_from_geuvadis.py   --vcf ${TRAINVCF} --rpkm ${TRAINRPKM} --gtf ${TRAINGTF} --chrom ${CHROM} --params 0.01 0.0 0.01 0.001 0.001 --outdir ${MODELDIR}
    python ${CODEBASE}/predict_on_gtex.py --vcf ${PREDVCF}  --model ${MODELDIR} --chrom ${CHROM} --outprefix predicted_gene_expression
