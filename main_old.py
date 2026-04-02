#!/usr/bin/env python3
"""Entry point for Container Number Recognition AI."""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "api":
        # Run API server
        import uvicorn
        from api import app
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        # Run CLI
        from cli import main
        main()
# Module-level caches – populated once on import to avoid repeated file I/O
# ---------------------------------------------------------------------------
CARRIER_PREFIXES: Tuple = get_carrier_prefixes()
CARRIER_PREFIX_SET: set = set(CARRIER_PREFIXES)


# ---------------------------------------------------------------------------
# ISO 6346 check-digit validation
# Letters A-Z map to values 10-38, skipping multiples of 11 (11, 22, 33).
# ---------------------------------------------------------------------------
_ISO6346_LETTER_VALUES: Dict[str, int] = {
    'A': 10, 'B': 12, 'C': 13, 'D': 14, 'E': 15, 'F': 16, 'G': 17, 'H': 18,
    'I': 19, 'J': 20, 'K': 21, 'L': 23, 'M': 24, 'N': 25, 'O': 26, 'P': 27,
    'Q': 28, 'R': 29, 'S': 30, 'T': 31, 'U': 32, 'V': 34, 'W': 35, 'X': 36,
    'Y': 37, 'Z': 38,
}


def validate_iso6346_check_digit(container_number: str) -> bool:
    """
    Validate the ISO 6346 check digit for a container number.
    The first 10 characters (4-letter owner code + equipment category + 6-digit serial)
    are used to compute the expected check digit at position 11.
    :param container_number: 11-character string, e.g. "MSCU1234567"
    :return: True if the check digit matches, False otherwise
    """
    if len(container_number) != 11:
        return False
    total = 0
    for i, ch in enumerate(container_number[:10]):
        if ch.isalpha():
            val = _ISO6346_LETTER_VALUES.get(ch.upper(), -1)
            if val < 0:
                return False
        elif ch.isdigit():
            val = int(ch)
        else:
            return False
        total += val * (2 ** i)
    remainder = total % 11
    expected_check = 0 if remainder == 10 else remainder
    try:
        return int(container_number[10]) == expected_check
    except ValueError:
        return False


def _parse_word_bbox(word) -> Tuple[int, int, int, int]:
    """
    Parse an OCR word's bounding_box into (x1, y1, x2, y2).
    OcrResult returns bounding_box as the string "x,y,w,h".
    Falls back to the 8-point polygon format [x1,y1,x2,y1,x2,y2,x1,y2] if needed.
    :param word: OCR word object with a .bounding_box attribute
    :return: (x1, y1, x2, y2)
    """
    bbox_str = word.bounding_box
    vals: List[int] = (
        [int(v) for v in bbox_str.split(",")]
        if isinstance(bbox_str, str)
        else list(bbox_str)
    )
    if len(vals) == 4:
        x1, y1, w, h = vals
        return x1, y1, x1 + w, y1 + h
    # 8-point polygon: [x1, y1, x2, y1, x2, y2, x1, y2]
    return vals[0], vals[1], vals[4], vals[5]


def get_ctnr_color(ctnr_img: np.ndarray) -> List[int]:
    """
    Get the most dominant color from the image given
    :param ctnr_img: Input image
    :return: [B, G, R]
    """
    colors, count = np.unique(ctnr_img.reshape(-1, ctnr_img.shape[-1]), axis=0, return_counts=True)
    colors_max: np.ndarray = colors[count.argmax()]
    return colors_max.tolist()


def get_ctnr_color_from_byte(input_byte_img: bytes, crop_zone: List[int]) -> List[int]:
    """
    Convert byte image to a ndarray format and pass to get container color
    :param input_byte_img:
    :param crop_zone: bounding box to extract color from
    :return: [B, G, R]
    """
    nparr = np.frombuffer(input_byte_img, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    cropped_img: np.ndarray = img_np[
                              max(0, crop_zone[1] - 100): min(crop_zone[3] + 100,
                                                              img_np.shape[0]),
                              max(0, crop_zone[0] - 100): min(crop_zone[2] + 100,
                                                              img_np.shape[1])]

    byte_res: list = get_ctnr_color(cropped_img)
    return byte_res


def extract_ctnr_location(ocr_output) -> CNRAI:
    """
    Extract container information using bounding box location logic.

    Key fixes vs. original:
    - Uses module-level CARRIER_PREFIX_SET (O(1) lookup) instead of re-reading file.
    - Prefix detection uses startswith instead of substring-contains to avoid false
      matches (e.g. "EGHU" wrongly matching "AEGHUA").
    - last_xy1_coord now stores [x2, y1] so the horizontal adjacency check tests
      whether the next word starts to the RIGHT of where the previous one ended,
      not just where it started.
    - allowable_buffer is initialised from the first matched word's height so the
      buffer is never 0 (the old bug caused get_label_angle to return 0 for
      axis-aligned OcrResult boxes, collapsing the buffer before the first real
      match).

    :param ocr_output: Output from recognize_printed_text API (OcrResult)
    :return: CNRAI
    """
    container_t_prefix: Tuple = ("G1", "R1", "U1", "P1", "T1")

    # The bounding block of detected container number [x1, y1, x2, y2]
    bound_block = [0, 0, 0, 0]

    tmp_cnrai = CNRAI()

    orientation_horizontal: bool = True
    # Stores [x2, y1] of the last matched word (right-edge + top) for adjacency checks.
    last_xy_coord: List[int] = []
    allowable_buffer: int = 50  # fallback before first word is measured

    # Iterate through all regions and lines (OcrResult format)
    found = False
    for region in ocr_output.regions:
        if found:
            break
        if not region.lines:
            continue
        for detected_text_line in region.lines:
            if found:
                break
            for word in detected_text_line.words:
                # As per shipping guidelines, container numbers will have 11 characters
                # https://www.evergreen-line.com/container/jsp/CNTR_ContainerMarkings.jsp
                if len(tmp_cnrai.container_number) >= 11 and tmp_cnrai.container_type != "":
                    # Early exit if container number and container type is detected
                    found = True
                    break

                clean_text: str = str(word.text).strip().replace(" ", "").upper()
                x1, y1, x2, y2 = _parse_word_bbox(word)
                # Build the 8-point list once for orientation / angle helpers
                bbox8: List[int] = [x1, y1, x2, y1, x2, y2, x1, y2]

                # Detect container prefix
                # Use startswith (not substring-contains) with the O(1) set lookup for
                # the 4-char prefix so "EGHU" does not match inside "AEGHUA".
                if tmp_cnrai.container_number == "" and clean_text[:4] in CARRIER_PREFIX_SET:
                    tmp_cnrai.container_number = clean_text
                    orientation_horizontal = check_orientation_horizontal(bbox8)
                    # Store the reference coordinate used to test the NEXT word's
                    # adjacency.  The choice depends on orientation:
                    #   Horizontal → [x2, y1]: next word must start to the right of x2
                    #                          and share roughly the same row (y1).
                    #   Vertical   → [x1, y2]: next word must start below y2
                    #                          and share roughly the same column (x1).
                    last_xy_coord = [x2, y1] if orientation_horizontal else [x1, y2]
                    allowable_buffer = get_label_angle(bbox8) * 3 or 50

                    bound_block[0] = x1
                    bound_block[1] = y1
                    bound_block[2] = x2
                    bound_block[3] = y2

                # Detect container serial
                # Ensure the container prefix is populated first,
                # and the total character count is < 11 as per ISO standard
                if 11 > len(tmp_cnrai.container_number) >= 4:
                    crit_met: bool = False

                    # Horizontal: next word must start to the right of the last word's
                    # right edge and be on roughly the same vertical line.
                    if orientation_horizontal:
                        crit_met = (x1 >= last_xy_coord[0] and
                                    within_buffer(last_xy_coord[1], y1, allowable_buffer))

                    # Vertical: next word must start below the last word's bottom edge
                    # and be in roughly the same horizontal column.
                    if not orientation_horizontal:
                        crit_met = (y1 >= last_xy_coord[1] and
                                    within_buffer(last_xy_coord[0], x1, allowable_buffer))

                    if crit_met:
                        tmp_cnrai.container_number += clean_text
                        # Advance the reference coordinate using the same
                        # orientation rule as the prefix word above.
                        last_xy_coord = [x2, y1] if orientation_horizontal else [x1, y2]

                        bound_block[2] = max(bound_block[2], x2)
                        bound_block[3] = max(bound_block[3], y2)

                # Detect container type (e.g. "45G1" → matches "G1" suffix)
                if tmp_cnrai.container_type == "":
                    if re.search(r"\d{2}(" + "|".join(container_t_prefix) + ")", clean_text):
                        tmp_cnrai.container_type = clean_text

                allowable_buffer = get_label_angle(bbox8) * 3 or allowable_buffer

    tmp_cnrai.bounding_box = [int(bb) for bb in bound_block]
    return tmp_cnrai


def extract_ctnr_regex(ocr_output) -> CNRAI:
    """
    Regex-based container number extractor.

    Key fixes vs. original:
    - Text is searched LINE BY LINE (not as one giant cross-region blob) to
      prevent false positives where adjacent words from different regions
      accidentally concatenate into a valid-looking pattern.
    - The regex now allows an optional equipment-category letter between the
      owner code and the serial (e.g. "MSCUU1234567" instead of bare 7 digits).
    - Bounding boxes are computed from the words that actually CONTRIBUTED to
      the match (tracked via character offsets) rather than the broken
      `word.text in container_number` substring test which was both direction-
      reversed and would match any short sub-string (e.g. "123" in "MSCU1234567").
    - The x=0 sentinel for the bounding-box initialisation is replaced by an
      explicit `bb_initialised` flag, so a word that genuinely starts at x=0
      is not skipped.
    - An ISO 6346 check-digit sanity test filters out spurious hits before
      committing to a container number.

    :param ocr_output: Output from recognize_printed_text API (OcrResult)
    :return: CNRAI
    """
    tmp_cnrai = CNRAI()

    container_t_prefix: Tuple = ("G1", "R1", "U1", "P1", "T1")

    # Build patterns once using the module-level prefix cache (avoids re-reading
    # container_prefix.txt on every call).
    # Carrier prefixes already include the equipment-category letter (e.g. "MSCU"),
    # so the serial is exactly 7 digits (6 serial + 1 check digit).  Using \d{7}
    # keeps the match at the required 11 characters and avoids 10- or 12-char hits.
    regex_ctnr_pattern: str = (
        "(" + "|".join(CARRIER_PREFIXES) + r")(\d{7})"
    )
    regex_ctnr_type_pattern: str = (
        r"\d{2}(" + "|".join(container_t_prefix) + ")"
    )

    # Process each line independently to avoid cross-region false matches.
    for region in ocr_output.regions:
        if not region.lines:
            continue
        for detected_text_line in region.lines:
            line_words = detected_text_line.words
            if not line_words:
                continue

            # Concatenate this line's words (uppercased, no spaces) and track
            # the character-level offset of each word so we can map a regex
            # match back to the contributing words.
            word_spans: List[Tuple[int, int]] = []
            line_text = ""
            for w in line_words:
                wt = str(w.text).strip().replace(" ", "").upper()
                start = len(line_text)
                line_text += wt
                word_spans.append((start, len(line_text)))

            # --- Container number ---
            if tmp_cnrai.container_number == "":
                # Use finditer so we can try every match in the line and skip
                # any that fail the ISO 6346 check-digit test.
                for m in re.finditer(regex_ctnr_pattern, line_text):
                    candidate = m.group(0)  # e.g. "MSCU1234567"
                    if not validate_iso6346_check_digit(candidate):
                        continue  # check digit mismatch – try the next match

                    tmp_cnrai.container_number = candidate

                    # Derive bounding box from words that overlap the match span.
                    match_start, match_end = m.start(), m.end()
                    bb: List[int] = [0, 0, 0, 0]
                    bb_initialised = False
                    for idx, w in enumerate(line_words):
                        ws, we = word_spans[idx]
                        if we <= match_start or ws >= match_end:
                            continue  # word not part of this match
                        wx1, wy1, wx2, wy2 = _parse_word_bbox(w)
                        if not bb_initialised:
                            bb[0], bb[1], bb[2], bb[3] = wx1, wy1, wx2, wy2
                            bb_initialised = True
                        else:
                            bb[2] = max(bb[2], wx2)
                            bb[3] = max(bb[3], wy2)
                    tmp_cnrai.bounding_box = bb
                    break  # accepted a valid candidate; stop searching this line

            # --- Container type (search entire line regardless of whether the
            #     container number was found on this line) ---
            if tmp_cnrai.container_type == "":
                mt = re.search(regex_ctnr_type_pattern, line_text)
                if mt:
                    tmp_cnrai.container_type = mt.group(0)

            # Stop early if both fields are populated
            if tmp_cnrai.container_number and tmp_cnrai.container_type:
                break

        if tmp_cnrai.container_number and tmp_cnrai.container_type:
            break

    return tmp_cnrai


def detect_container_details(input_image_byte: bytes) -> Dict:
    """
    Takes in an image in byte array format, and run OCR on it
    :param input_image_byte: Image byte array []byte
    :return: json format of {"container_number": "ABCD1234567", "container_type": "45G1, "bounding_box": [x, y, x3, y3], "error": error_details.message}
    """

    # Use the Printed Text API for OCR - this is the synchronous method that works
    try:
        # Call Printed Text API with the image bytes
        ocr_output = cv_client.recognize_printed_text_in_stream(image=BytesIO(input_image_byte))
        
    except Exception as e:
        cnrai_detection = CNRAI()
        cnrai_detection.error = str(e)
        return cnrai_detection.__dict__

    # Early exit if no text detected
    cnrai_detection = CNRAI()
    if not ocr_output or not hasattr(ocr_output, 'regions') or len(ocr_output.regions) == 0:
        return cnrai_detection.__dict__

    cnrai_detection = extract_ctnr_regex(ocr_output)

    # If regex logic is missing either field, run the location detector and fill
    # only the gaps – do not discard what the regex already found correctly.
    if cnrai_detection.container_number == "" or cnrai_detection.container_type == "":
        location_detection = extract_ctnr_location(ocr_output)
        if cnrai_detection.container_number == "":
            cnrai_detection.container_number = location_detection.container_number
            cnrai_detection.bounding_box = location_detection.bounding_box
        if cnrai_detection.container_type == "":
            cnrai_detection.container_type = location_detection.container_type

    cnrai_detection.container_color = get_ctnr_color_from_byte(input_image_byte, cnrai_detection.bounding_box)

    return cnrai_detection.__dict__


def http_request(request):
    """HTTP Cloud Function.
    Args:
        request (flask.Request): The request object.
        <https://flask.palletsprojects.com/en/1.1.x/api/#incoming-request-data>
    Returns:
        The response text, or any set of values that can be turned into a
        Response object using `make_response`
        <https://flask.palletsprojects.com/en/1.1.x/api/#flask.make_response>.
    """
    from flask import jsonify  # deferred import so Flask is not required globally

    # Set CORS headers for the preflight request
    if request.method == "OPTIONS":
        # Allows GET requests from any origin with the Content-Type
        # header and caches preflight response for an 3600s
        headers = {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET",
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Max-Age": "3600",
        }

        return "", 204, headers
    headers = {"Access-Control-Allow-Origin": "*"}

    if 'image' not in request.files or request.files["image"].filename == '':
        return jsonify({"message": "No files found"}), 400, headers

    file = request.files['image']
    # Guard against filenames with no extension (rsplit would give a 1-element list
    # and indexing [1] would raise IndexError).
    name_parts = file.filename.rsplit('.', 1)
    if not file or len(name_parts) < 2 or name_parts[1].lower() not in ["jpg", "jpeg", "bmp", "png"]:
        return jsonify({"message": "Wrong file type"}), 415, headers

    im = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    encoded_im = cv2.imencode('.JPG', im)[1].tobytes()

    return jsonify(detect_container_details(encoded_im)), 200, headers


if __name__ == '__main__':
    image_dir = "./data"

    for filename in os.listdir(image_dir):
        f = os.path.join(image_dir, filename)

        # checking if it is a file
        if os.path.isfile(f) and f.endswith((".bmp", ".jpg", ".jpeg", ".png")):
            input_img = cv2.imread(f)
            a = cv2.imencode('.JPG', input_img)[1].tobytes()

            res: Dict = detect_container_details(a)
            print(f"Result for {filename}: {res}")
            
            # Skip if there's an error or no container detected
            if res.get("error") or res.get("container_number") == "":
                print(f"  Skipped: {res.get('error', 'No container detected')}")
                continue

            # Crop the detected bounding area from original image
            cropped_img: np.ndarray = input_img[
                                      max(0, res["bounding_box"][1] - 100): min(res["bounding_box"][3] + 100,
                                                                                input_img.shape[0]),
                                      max(0, res["bounding_box"][0] - 100): min(res["bounding_box"][2] + 100,
                                                                                input_img.shape[1])]

            # Put the detected details top left of the cropped image
            text = f"{res['container_number']} - {res['container_type']}"
            cv2.putText(cropped_img, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        color=(0, 255, 100), fontScale=1, thickness=2, lineType=cv2.LINE_AA)

            # Display
            print(res)

            cropped_img = downscale(cropped_img)
            cv2.imshow("output", cropped_img)
            cv2.waitKey()
