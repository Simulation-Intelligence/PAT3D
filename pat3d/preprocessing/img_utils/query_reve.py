import requests
import time
from urllib.parse import urljoin



def query_reve_img(prompt, api_key, img_num):

    url = "https://reveapi.com/api/generate-image"
    headers = {
        "Content-Type": "application/json",
        "Authorization": api_key
    }
    data = {
        "prompt": prompt, 
        "style": "photorealistic",
        "width": 1024,
        "height": 1024
    }

    response = requests.post(url, headers=headers, json=data)

    return response.json()

def query_status(reve_response, api_key):

    base_url = "https://reveapi.com"  # Base URL for the API
    status_url = base_url + reve_response['status_url']
    result_url = base_url + reve_response['result_url']
    headers = {
        "Authorization": f"Bearer {api_key}"
    }


    # Polling the status URL
    while True:
        response = requests.get(status_url, headers=headers)

        if response.status_code != 200:
            time.sleep(2)  # Wait before retrying
            continue
        
        status_data = response.json()
        if status_data.get('status') == 'succeeded':
            #print("Image generation completed!")
            break
        elif status_data.get('status') == 'processing':
            print("Still processing...")
            time.sleep(2)  # Wait 5 seconds before polling again
        else:
            print("Image generation failed.")
            return None

    return result_url

def fetch_img_urls(result_url, api_key):
    """
    Downloads the image file from the result URL and saves it to a local file.

    Args:
        result_url (str): The URL to fetch the image from.
        api_key (str): The API key for authentication.
        output_file (str): The path to save the downloaded image file.

    Returns:
        bool: True if the image is successfully downloaded and saved, False otherwise.
    """
    base_url = "https://reveapi.com"  # Base URL for the API
    full_url = urljoin(base_url, result_url)
    headers = {
        "Authorization": f"Bearer {api_key}"
    }

    try:
        response = requests.get(full_url, headers=headers, stream=True)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Failed to download image. Status code: {response.status_code}")
            return None
    
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while downloading the image: {e}")
        return None

def download_image(image_url, output_file):
    #output_file = "debug.png"
    """
    Downloads an image from the given URL and saves it to a local file.

    Args:
        image_url (str): The URL of the image to download.
        output_file (str): The path to save the downloaded image file.

    Returns:
        bool: True if the image is successfully downloaded and saved, False otherwise.
    """
    try:
        # Send GET request to the image URL
        response = requests.get(image_url, stream=True)
        if response.status_code == 200:
            # Save the image content to a file
            with open(output_file, 'wb') as file:
                for chunk in response.iter_content(1024):
                    file.write(chunk)
            print(f"Image successfully downloaded and saved to {output_file}")
            return True
        else:
            print(f"Failed to download image. Status code: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while downloading the image: {e}")
        return False
    

def load_apikey(api_path):
    with open(api_path, 'r') as file:
        api_key = file.read().strip()
    return api_key


