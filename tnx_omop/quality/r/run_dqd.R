# Loads a tnx-omop output folder into DuckDB and runs the OHDSI Data Quality Dashboard.
#
# Called by `tnx-omop dqd` (tnx_omop/quality/dqd.py):
#   Rscript run_dqd.R <database> <results_json> <source_name> <keep_database> <table>=<parquet> ...
# Exit status 3 when an R package is missing (install_packages.R installs them),
# 1 on any other error.

required <- c("duckdb", "DatabaseConnector", "CommonDataModel", "DataQualityDashboard")
installed <- vapply(required, requireNamespace, logical(1), quietly = TRUE)
if (!all(installed)) {
  message("Missing R packages: ", paste(required[!installed], collapse = ", "))
  quit(status = 3)
}

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 5) {
  stop("usage: run_dqd.R <database> <results_json> <source_name> <keep_database> <table>=<parquet> ...")
}
database <- normalizePath(args[1], winslash = "/", mustWork = FALSE)
results_json <- normalizePath(args[2], winslash = "/", mustWork = FALSE)
source_name <- args[3]
keep_database <- identical(args[4], "true")
pairs <- args[-(1:4)]
tables <- sub("=.*$", "", pairs)
paths <- normalizePath(sub("^[^=]*=", "", pairs), winslash = "/", mustWork = TRUE)

# duckdb (R package 1.5.6, Windows) crashes the R process when it autoloads the ICU
# extension in the middle of a session, which the CURRENT_DATE of the plausibleValueHigh
# checks triggers. Loading ICU right after each connect avoids it. DQD opens its own
# connections, so the load is added to DatabaseConnector::connect with trace().
# DatabaseConnector installs ICU on connect (a download); when that fails, the load
# only warns.
trace(
  "connect",
  where = asNamespace("DatabaseConnector"),
  print = FALSE,
  exit = quote(tryCatch(
    DatabaseConnector::executeSql(
      returnValue(), "LOAD icu;", progressBar = FALSE, reportOverallTime = FALSE
    ),
    error = function(e) warning("Cannot load the DuckDB ICU extension: ", conditionMessage(e))
  ))
)

remove_database <- function() {
  unlink(c(database, paste0(database, ".wal")))
}
remove_database()
details <- DatabaseConnector::createConnectionDetails(dbms = "duckdb", server = database)

# All CDM 5.4 tables, also those the converter does not write: DQD checks every table.
# Primary and foreign keys are left out: DQD checks key uniqueness and references
# itself, and reports them per field instead of stopping the load at the first one.
CommonDataModel::executeDdl(
  details,
  cdmVersion = "5.4",
  cdmDatabaseSchema = "main",
  executeDdl = TRUE,
  executePrimaryKey = FALSE,
  executeForeignKey = FALSE
)

connection <- DatabaseConnector::connect(details)
for (i in seq_along(tables)) {
  # BY NAME matches columns by name; a Parquet column that is not in the CDM table fails.
  sql <- sprintf(
    "INSERT INTO main.%s BY NAME SELECT * FROM read_parquet('%s');",
    tables[i], gsub("'", "''", paths[i], fixed = TRUE)
  )
  DatabaseConnector::executeSql(connection, sql, progressBar = FALSE, reportOverallTime = FALSE)
  rows <- DatabaseConnector::querySql(connection, sprintf("SELECT COUNT(*) AS n FROM main.%s;", tables[i]))
  message(sprintf("Loaded %s: %s rows", tables[i], format(rows[[1]], big.mark = ",")))
}
DatabaseConnector::disconnect(connection)

DataQualityDashboard::executeDqChecks(
  connectionDetails = details,
  cdmDatabaseSchema = "main",
  resultsDatabaseSchema = "main",
  vocabDatabaseSchema = "main",
  cdmSourceName = source_name,
  cdmVersion = "5.4",
  numThreads = 1,
  outputFolder = dirname(results_json),
  outputFile = basename(results_json),
  writeToTable = FALSE
)

if (!keep_database) {
  remove_database()
}
