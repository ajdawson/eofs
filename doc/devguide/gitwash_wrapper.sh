#!/bin/bash
#
# Run gitwash_dumper.py and edit the output to remove the email patching
# page which is not relevant to this project.
#

set -u
set -e

readonly PROJECT_NAME="eofs"
readonly REPO_NAME="eofs"
readonly GITHUB_USER="ajdawson"
readonly PROJECT_URL="http://ajdawson.github.io/eofs"
readonly OUTPUT_DIRECTORY="./"
readonly GITWASH_DUMPER="./gitwash_dumper.py"

# Use the gitwash script to refresh the documentation.
python "$GITWASH_DUMPER" "$OUTPUT_DIRECTORY" "$PROJECT_NAME" \
                         --repo-name="$REPO_NAME" \
                         --github-user="$GITHUB_USER" \
                         --project-url="$PROJECT_URL" \
                         --project-ml-url="NONE"

# Remove the patching section of the gitwash guide.
rm -f "${OUTPUT_DIRECTORY}/gitwash/patching.rst"
sed -i '/patching/d' "${OUTPUT_DIRECTORY}/gitwash/index.rst"

# Remove references to the project mailing list in the gitwash guide.
sed -i '/mailing list/d' "${OUTPUT_DIRECTORY}/gitwash/this_project.inc"
sed -i '/mailing list/d' "${OUTPUT_DIRECTORY}/gitwash/development_workflow.rst"

# Remove all trailing whitespace and trailing blank lines from the downloaded
# gitwash guide restructured text sources:
sed -i 's/[[:space:]]*$//' "${OUTPUT_DIRECTORY}"/gitwash/*.{rst,inc}
sed -i -e :a -e '/^\n*$/{$d;N;ba' -e '}' "${OUTPUT_DIRECTORY}"/gitwash/*.{rst,inc}
