import cv2
import numpy as np

# Initialize camera
cap = cv2.VideoCapture(0)

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    #titik koordinat diatur agar mudah untuk di warp
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # P1
    rect[2] = pts[np.argmax(s)]  # P2
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # P3
    rect[3] = pts[np.argmax(diff)]  # P4
    
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    return warped

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Preprocess frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Initialize card counter
    card_count = 0
    
    # Filter contours
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:  # Adjust the threshold based on your needs
            # Find corners of the contour
            approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
            if len(approx) == 4:
                #mencari titik koordinat siku kartu dan meletakkannya ke corners
                corners = []
                for corner in approx:
                    x, y = corner.ravel()
                    corners.append((x, y))
                    cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
                corners = np.array(corners, dtype="float32")
                warped = four_point_transform(frame, corners)
                
                # Display the warped card in a new window
                window_name = f'Warped Card {card_count + 1}'
                cv2.imshow(window_name, warped)
                card_count += 1
    
    # Display the resulting frame
    cv2.imshow('Card Detection', frame)
    
    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
