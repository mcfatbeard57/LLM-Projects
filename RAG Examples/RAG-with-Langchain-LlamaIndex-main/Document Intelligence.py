import logging
import requests
import time
from typing import Union, Dict
from config.settings import get_settings


class DocumentIntelligenceService:
    """
    A service class for interacting with Azure Document Intelligence API.
    This class provides methods to analyze documents using Azure's Document Intelligence service.
    """

    def __init__(self):
        """
        Initialize the DocumentIntelligenceService with API credentials and endpoint.
        """
        settings = get_settings()
        self.key = settings.document_intelligence.api_key
        self.endpoint = settings.document_intelligence.endpoint
        self.api_version = "2024-02-29-preview"  # Currently only available in East US, West US2, and West Europe

    def analyze(
        self,
        source: Union[str, bytes],
        is_url: bool = True,
        model_id: str = "prebuilt-layout",
    ) -> Dict:
        """
        Analyze a document using Azure Document Intelligence.

        Args:
            source (Union[str, bytes]): The document source, either a URL or base64 encoded content.
            is_url (bool): True if the source is a URL, False if it's base64 encoded content.
            model_id (str): The ID of the model to use for analysis.

        Returns:
            Dict: The analysis results.

        Raises:
            requests.HTTPError: If the API request fails.
        """
        result_id = self._submit_analysis(source, is_url, model_id)
        return self._get_analysis_results(result_id, model_id)

    def _submit_analysis(
        self, source: Union[str, bytes], is_url: bool, model_id: str
    ) -> str:
        """
        Submit a document for analysis to Azure Document Intelligence.

        Args:
            source (Union[str, bytes]): The document source, either a URL or base64 encoded content.
            is_url (bool): True if the source is a URL, False if it's base64 encoded content.
            model_id (str): The ID of the model to use for analysis.

        Returns:
            str: The result ID for the submitted analysis.

        Raises:
            ValueError: If the Operation-Location header is missing in the response.
            requests.HTTPError: If the API request fails.
        """
        url = f"{self.endpoint}/documentintelligence/documentModels/{model_id}:analyze?api-version={self.api_version}&outputContentFormat=markdown"
        headers = {
            "Content-Type": "application/json",
            "Ocp-Apim-Subscription-Key": self.key,
        }
        data = {"urlSource": source} if is_url else {"base64Source": source}

        logging.info("Submitting document for analysis")
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()

        operation_location = response.headers.get("Operation-Location")
        if not operation_location:
            raise ValueError("Operation-Location header is missing in the response.")

        return operation_location.split("/")[-1].split("?")[0]

    def _get_analysis_results(self, result_id: str, model_id: str) -> Dict:
        """
        Retrieve the analysis results from Azure Document Intelligence.

        Args:
            result_id (str): The ID of the analysis result to retrieve.
            model_id (str): The ID of the model used for analysis.

        Returns:
            Dict: The analysis results.

        Raises:
            requests.HTTPError: If the API request fails.
        """
        url = f"{self.endpoint}/documentintelligence/documentModels/{model_id}/analyzeResults/{result_id}?api-version={self.api_version}&outputContentFormat=markdown"
        headers = {"Ocp-Apim-Subscription-Key": self.key}

        while True:
            logging.info("Waiting for analysis to complete.")
            time.sleep(2)
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()

            if data.get("status") in ["succeeded", "failed"]:
                return data


if __name__ == "__main__":
    # Example usage of the DocumentIntelligenceService
    client = DocumentIntelligenceService()
    analysis_results = client.analyze(
        source="https://s2.q4cdn.com/299287126/files/doc_financials/2024/ar/Amazon-com-Inc-2023-Annual-Report.pdf"
    )
    print(analysis_results.keys())
    print(analysis_results["analyzeResult"].keys())
    print(analysis_results["analyzeResult"]["content"])
    print(analysis_results["analyzeResult"]["tables"])
