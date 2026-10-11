#!/bin/bash
# Downloads SEDA 6.0 (Stanford Digital Repository, druid xh833nn4025; world-downloadable) into Datastore/District Covariates/seda
cd "$(dirname "$0")/../Datastore/District Covariates/seda" || exit 1
U=https://stacks.stanford.edu/file/xh833nn4025
for f in seda_admindist_pool_cs_6.0.csv seda_admindist_poolsub_cs_6.0.csv seda_admindist_annual_cs_6.0.csv \
         seda_cov_admindist_pool_6.0.csv seda_cov_admindist_annual_6.0.csv \
         seda_county_pool_cs_6.0.csv seda_cov_county_pool_6.0.csv seda_cov_county_annual_6.0.csv; do
  [ -s "$f" ] || curl -sL --retry 3 -C - -o "$f" "$U/$f" -w "$f %{http_code} %{size_download}\n"
done
