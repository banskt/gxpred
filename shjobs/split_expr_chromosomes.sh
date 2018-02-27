

HOME='/usr/users/fsimone'
ENV="${HOME}/myenv"
GTFPATH="${HOME}/datasets/gtex/gencode.v19.annotation.gtf.gz"
EXPR="${HOME}/datasets/gtex/gtex_wholeblood_normalized.expression.txt"
OUT="${HOME}/datasets/gtex/expression"

for CHR in `seq 1 22`
do

${ENV}/bin/python split_gene_expr_in_chromosomes.py --gtf ${GTFPATH} \
													--expr ${EXPR} --chr ${CHR} \
													--outdir ${OUT}
done