# Installs the R packages that `tnx-omop dqd` runs, at the versions run_dqd.R was
# written against. DataQualityDashboard and CommonDataModel need rJava, so a Java JDK
# must be installed and found by R (JAVA_HOME) even though DuckDB itself needs no Java.
#   Rscript install_packages.R

options(repos = c(CRAN = "https://cloud.r-project.org"))
if (!requireNamespace("remotes", quietly = TRUE)) {
  install.packages("remotes")
}
install.packages("duckdb")
remotes::install_github("OHDSI/DatabaseConnector@v7.2.0", upgrade = "never")
remotes::install_github("OHDSI/CommonDataModel@v5.5.0", upgrade = "never")
remotes::install_github("OHDSI/DataQualityDashboard@v2.8.9", upgrade = "never")

# install.packages and install_github only warn when a package fails to install
required <- c("duckdb", "DatabaseConnector", "CommonDataModel", "DataQualityDashboard")
loaded <- vapply(required, requireNamespace, logical(1), quietly = TRUE)
if (!all(loaded)) {
  message("Not installed or not loadable: ", paste(required[!loaded], collapse = ", "))
  quit(status = 1)
}
