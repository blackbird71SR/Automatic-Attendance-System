import face_recognition
import cv2
import os


def extract_faces():
    i=0
    for eachImg in os.listdir("temp"):
        # print(eachImg, 'read')
        try:
            img = face_recognition.load_image_file("temp/" + eachImg)
        except PermissionError:
            continue
        face_locations = face_recognition.face_locations(img)
        for(top,right,bottom,left) in face_locations:
            sub_face = img[top:bottom, left:right]
            face_file_name = "temp1/presentFaces/face_" + str(i) + ".jpg"
            cv2.imwrite(face_file_name,sub_face)
            i=i+1
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print (i, 'faces read')

def match():
    present = {}
    known_faces = []
    known_faces_names = []
    path = "registration_images/Year2"
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    for imagePath in imagePaths:
        img = face_recognition.load_image_file(imagePath)
        known_faces_names.append(imagePath.split(".")[0].split("\\")[1])
        known_faces.append(face_recognition.face_encodings(img)[0])
    # print(known_faces_names)
    path = "temp1/presentFaces"
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    for imagePath in imagePaths:
        img = face_recognition.load_image_file(imagePath)
        try:
            unknown_face_encoading = face_recognition.face_encodings(img)[0]
        except IndexError:
            continue
        print(imagePath," Read...")
        results = face_recognition.compare_faces(known_faces, unknown_face_encoading)
        indices = [i for i, x in enumerate(results) if x == True]
        for each in indices:
            present[known_faces_names[each]] = "Present"
    print(present)

extract_faces()
match()