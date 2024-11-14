import requests

BASE_URL = "http://127.0.0.1:8000"

def test_correct_text_api():
    payload = {
        "text": "Madrid eres una ciudad hermosa.",
        "lang_code": "es_XX"
    }
    
    # Make a POST request to the API
    response = requests.post(f"{BASE_URL}/api", json=payload)
    
    # Assert that the response status code is 200 (OK)
    assert response.status_code == 200, f"Expected status code 200, got {response.status_code}"
    
    # Parse the JSON response
    data = response.json()
    
    # Display the corrected text and corrections
    print("Corrected Text:", data["corrected_text"])
    print("Corrections:", data["corrections"])

    return data

result = test_correct_text_api()
print(result)
