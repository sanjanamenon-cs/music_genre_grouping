import firebase_admin
from firebase_admin import credentials, firestore

cred= credentials.Certificate("C:\Users\admin\Downloads\ml_music_genre_grouping-main\music-95847-firebase-adminsdk-fbsvc-304e25522e.json")
firebase_admin.initialize_app(cred)
db = firestore.client()
print("Firebase setup complete.")

doc_ref = db.collection(u'music_data').document(u'example_doc')
doc_ref.set({
    "name": "Sanjana",
    "email": menon.s.sanjana@gmail.com,
})
print("Document written to Firestore.")
