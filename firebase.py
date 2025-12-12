import firebase_admin
from firebase_admin import credentials, firestore

# Initialize Firebase (uses your existing serviceAccountKey.json)
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)

db = firestore.client()

# --- Demo: add and read a user ---
demo_user = {"name": "Sanjana", "role": "demo"}

# Add user to Firestore
db.collection("demo_users").document("user1").set(demo_user)
print("Added demo user to Firebase!")

# Read user from Firestore
doc = db.collection("demo_users").document("user1").get()
if doc.exists:
    print("Read from Firebase:", doc.to_dict())
else:
    print("User not found in Firebase.")