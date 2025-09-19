# script that takes the name of a csv file and corrects it: skips the header and
# then join every two lines. then replace the square brackets element
# "...,[ ... ], ..." with the custom value 0.0001 and output the file to stdout
# without modifying the original file.
# Usage: ./correct_csv.sh <csv_file>

# for every line:
while IFS= read -r line; do
    # skip the header
    if [[ $line == "seed"* ]]; then
        echo "$line"
        continue
    fi
    # join every two lines
    if [[ $line == *"["* ]]; then
        read -r next_line
        line="${line}${next_line}"
        # replace the square brackets element with 0.0001
        line=$(echo "$line" | sed 's/\[.*\]/0.0001/g')
        echo "$line"
    fi
done < "$1"
