


ENV='/usr/users/fsimone/myenv'
GTFPATH="/usr/users/fsimone/datasets/gtex/gencode.v19.annotation.gtf.gz"
EXPR="/usr/users/fsimone/datasets/gtex/gtex_wholeblood_normalized.expression.txt"
OUT="/usr/users/fsimone/datasets/gtex/expression"

for CHR in `seq 1 1`
do

${ENV}/bin/python split_gene_expr_in_chromosomes.py --gtf ${GTFPATH} \
													--expr ${EXPR} --chr ${CHR} \
													--outdir ${OUT}
done