from google import genai

client = genai.Client(api_key="AQ.Ab8RN6L-KFFNOx-KNr_C07cU81JAS4uBMRvBM67gNWaUke_JKw")

response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="Say hello in one line"
)

print(response.text)