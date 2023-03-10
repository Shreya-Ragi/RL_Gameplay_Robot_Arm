import cv2
import numpy as np
from collections import defaultdict


def edge_detection(image, l_thresh = 50, u_thresh = 150):
    edges = cv2.Canny(image, l_thresh, u_thresh)

    return edges


def hough_lines(edges, image):
    lines = cv2.HoughLines(edges,1,np.pi/180,200)
    for line in lines:
        rho,theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv2.line(image,(x1,y1),(x2,y2),(0,255,0),2)

    return lines


def segment_by_angle_kmeans(lines, k=2, **kwargs):
    # Define criteria = (type, max_iter, epsilon)
    default_criteria_type = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER
    criteria = kwargs.get('criteria', (default_criteria_type, 10, 1.0))

    # returns angles in [0, pi] in radians
    angles = np.array([line[0][1] for line in lines])

    # multiply the angles by two and find coordinates of that angle
    pts = np.array([[np.cos(2*angle), np.sin(2*angle)]
                    for angle in angles], dtype=np.float32)

    # run kmeans on the coords
    labels, centers = cv2.kmeans(pts, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)[1:]
    labels = labels.reshape(-1)  # transpose to row vec

    # segment lines based on their kmeans label
    segmented = defaultdict(list)
    for i, line in enumerate(lines):
        segmented[labels[i]].append(line)
    segmented = list(segmented.values())

    return segmented


def intersection(line1, line2):
    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    
    return [[x0, y0]]


def segmented_intersections(lines, image):
    intersections = []
    for i, group in enumerate(lines[:-1]):
        for next_group in lines[i+1:]:
            for line1 in group:
                for line2 in next_group:
                    intersections.append(intersection(line1, line2)) 
    
    for pt in intersections:
        x = pt[0][0]
        y = pt[0][1]
        image = cv2.circle(image, (x,y), radius=0, color=(0, 0, 255), thickness=6)

    return intersections


if __name__ == "__main__":
    image = cv2.imread("image.png")
    img_grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    cv2.imshow("grayscale", img_grey)
    cv2.waitKey(0)

    edges = edge_detection(img_grey)
    cv2.imshow("edges", edges)
    cv2.waitKey(0)

    lines = hough_lines(edges, image)
    cv2.imshow("lines", image)
    cv2.waitKey(0)

    segmented = segment_by_angle_kmeans(lines)
    intersections = segmented_intersections(segmented, image)
    cv2.imshow("intersections", image)
    cv2.waitKey(0)

    cv2.destroyAllWindows()