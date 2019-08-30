# emptyDrops analysis of unfiltered counts data for intelligent labeling of cell barcodes
# C Heiser, August 2019

rm(list=ls()) # clear workspace
suppressPackageStartupMessages(require(argparse))
suppressPackageStartupMessages(require(tidyverse))
suppressPackageStartupMessages(require(DropletUtils))


run.emptyDrops <- function(m, lower, FDR.thresh=0.01, ...){
  # wrapper for emptyDrops function
  #   m = counts dataframe in genes x cells format
  #   lower = number of total UMI counts below which all barcodes are considered empty droplets
  #   FDR.thresh = threshold above which barcodes are labeled as empty droplets (default 0.01)
  #   ... = additional args to pass to emptyDrops() function
  start.time <- Sys.time()
  
  result <- emptyDrops(m = m, lower = lower, ...)
  
  out <- data.frame(result@listData) %>%
    mutate(empty = as.numeric(FDR > FDR.thresh)) %>%
    mutate(empty = replace_na(empty, 1)) %>%
    mutate(barcode = result@rownames)
  
  print(Sys.time() - start.time)
  return(out)
}


read.counts <- function(path, transpose=F){
  start.time <- Sys.time()
  if(str_detect(string = path, pattern = '.csv')){
    counts <- read.csv(path)
  }else if(str_detect(string = path, pattern = '.tsv')){
    counts <- read.csv(path, sep = '\t')
  }
  
  if(transpose){
    counts <- t(counts)
  }
  print(Sys.time() - start.time) # print file reading time
  return(counts)
}



if(!interactive()){
  # create parser object
  parser <- ArgumentParser()
  
  # import options
  parser$add_argument('counts',
                      help='Path to counts matrix as .tsv or .csv file')
  parser$add_argument('-cxg', '--cellxgene', action='store_true', default=FALSE,
                      help='Counts file in cell x gene format (cells as rows)?')
  parser$add_argument('-l', '--lower', type='integer', default=500,
                      help='Total UMI cutoff for determining empty droplets')
  parser$add_argument('-fdr', '--fdrthresh', type='double', default=0.01,
                      help='Threshold for determining empty droplet from returned FDR value')
  parser$add_argument('-o', '--output', default='./emptydrops.csv',
                      help='Output location for emptyDrops analysis')
  
  # get command line options, if help encountered print help and exit,
  #   otherwise if options not found on command line, set defaults
  args <- parser$parse_args()
  
  # read counts data into df
  message('\nReading in file: ', args$counts)
  counts <- read.counts(path = args$counts, transpose = args$cellxgene)
  
  # perform emptyDrops analysis
  message('\nRunning emptyDrops...')
  out <- run.emptyDrops(m = as.matrix(counts), lower = as.numeric(args$lower), FDR.thresh = as.numeric(args$fdrthresh))
  
  # write to file
  message('\nWriting to file: ', args$output)
  write.csv(out, args$output, row.names=F)
  message('\nDone!')
}
