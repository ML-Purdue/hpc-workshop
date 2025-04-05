#!/usr/bin/bash

curl -LO https://dagshub.com/ML-Purdue/hpc-workshop/raw/c8264ae9bb608d4b3ebcc3c73f180f300a553bcc/s3:/hpc-workshop/data.tgz
tar xzf data.tgz
rm data.tgz
mkdir models
