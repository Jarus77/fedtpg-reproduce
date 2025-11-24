#!/bin/bash
# Extract results from pre-trained model log file

LOG_FILE="output/cross_cls/fedtpg/20_8/43/log.txt"

echo "================================"
echo "FedTPG Cross-Class Results"
echo "================================"
echo ""

echo "BASE Set Results (Seen Classes):"
echo "--------------------------------"

# Get the section between "base" and the final avg accuracy
tail -300 "$LOG_FILE" | sed -n '/Evaluate on the \*base\* set of caltech101/,/avg accuracy:/p' | grep "accuracy:" | head -9

echo ""
echo "NEW Set Results (Unseen Classes):"
echo "--------------------------------"

# Get the section for new set
tail -300 "$LOG_FILE" | sed -n '/Evaluate on the \*new\* set of caltech101/,/avg accuracy:/p' | grep "accuracy:" | head -9

echo ""
echo "Averages:"
echo "--------------------------------"
tail -300 "$LOG_FILE" | grep "avg accuracy:" | tail -2
