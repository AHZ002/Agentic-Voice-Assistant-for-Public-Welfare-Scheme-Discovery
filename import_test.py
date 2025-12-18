from memory.session_memory import SessionMemory
from memory.contradiction_handler import ContradictionHandler

from tools.eligibility import EligibilityEngine, UserProfile, EligibilityCriteria
from tools.scheme_retriever import SchemeRetriever
from tools.mock_gov_api import MockGovAPI

def main():
    memory = SessionMemory()
    checker = ContradictionHandler()
    engine = EligibilityEngine()
    retriever = SchemeRetriever()
    api = MockGovAPI()

    print("All imports OK")

if __name__ == "__main__":
    main()
