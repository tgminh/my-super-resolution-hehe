#!/bin/bash

OUTPUT_FILE="output2_480.txt"
SCRIPT="full2.py"
WEIGHTS="fsrcnn_x2.pth"
IMG="data/480/frame001_480.png"
REF="data/480/ref_frame001_480.png"
SCALE=2

# Clear previous output file
echo "PSNR and time results" > $OUTPUT_FILE
echo "=====================" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

for device in cuda cpu
do
    for tiles in {1..8}
    do
        echo "Running device=$device tiles=$tiles..."

        # Run and capture output
        RESULT=$(python "$SCRIPT" \
            --weights-file "$WEIGHTS" \
            --image-file "$IMG" \
            --scale "$SCALE" \
            --device "$device" \
            --reference-file "$REF" \
            --tiles "$tiles"
        )

        # Extract PSNR and Model (batched) time
        PSNR=$(echo "$RESULT" | grep "PSNR" | sed 's/PSNR: //')
        TIME=$(echo "$RESULT" | grep "Model (batched) time" | awk '{print $4, $5, $6}')  

        # Write to output2.txt
        echo "device=$device tiles=$tiles" >> $OUTPUT_FILE
        echo "PSNR: $PSNR" >> $OUTPUT_FILE
        echo "Model batched time: $TIME" >> $OUTPUT_FILE
        echo "" >> $OUTPUT_FILE
    done
done

echo "Done! Results saved to $OUTPUT_FILE"

#!/bin/bash

OUTPUT_FILE="output_480.txt"
SCRIPT="full.py"
WEIGHTS="fsrcnn_x2.pth"
IMG="data/480/frame001_480.png"
REF="data/480/ref_frame001_480.png"
SCALE=2

# Reset output file
echo "Inference Results" > "$OUTPUT_FILE"
echo "=================" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

for device in cuda cpu
do
    for tiles in {1..8}
    do
        echo "Running device=$device tiles=$tiles..."

        # Run the script and capture output
        RESULT=$(python "$SCRIPT" \
            --weights-file "$WEIGHTS" \
            --image-file "$IMG" \
            --scale "$SCALE" \
            --device "$device" \
            --reference-file "$REF" \
            --tiles "$tiles"
        )

        # Extract model batched time: from "[INFO] Total inference: 1010.78 ms"
        TIME=$(echo "$RESULT" | grep "Total inference:" | awk '{print $4, $5}')

        # Extract PSNR: from "[INFO] PSNR vs reference: 39.14 dB"
        PSNR=$(echo "$RESULT" | grep "PSNR vs reference" | awk '{print $5, $6}')

        # Append to output file
        echo "device=$device tiles=$tiles" >> "$OUTPUT_FILE"
        echo "Model batched time: $TIME" >> "$OUTPUT_FILE"
        echo "PSNR: $PSNR" >> "$OUTPUT_FILE"
        echo "" >> "$OUTPUT_FILE"
    done
done

echo "Done! Results saved to $OUTPUT_FILE"
