import os
import requests
import json
import base64
import traceback
from typing import List, Union, Generator, Iterator, Sequence
from pydantic import BaseModel, Field
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool, tool
from langchain_community.llms import Ollama
from llama_index.core import VectorStoreIndex, Settings, Document
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.llms import MockLLM

index = None
documents = None

class SearchRepositoryInput(BaseModel):
    query: str = Field(description="Search query on github repository files.")

@tool("search_repository", args_schema=SearchRepositoryInput, return_direct=False)
def search_repository(query: str) -> str:
    """Search GitHub repository files and return information based on the query."""
    try:
        global index, documents
        
        # Query the index
        query_engine = index.as_query_engine(llm=MockLLM())
        response = query_engine.query(query)
        return response.response

    except Exception as e:
        print(f"Error in search_repository: {str(e)}")
        return "An error occurred while searching the repository."

class Pipeline:

    class Valves(BaseModel):

        OPENAI_API_BASE_URL: str = "https://api.openai.com/v1"
        OPENAI_API_KEY: str = ""
        OPENAI_API_MODEL: str = "gpt-4o"
        OPENAI_API_TEMPERATURE: float = 0.7
        OPENAI_EMBED_MODEL: str = "text-embedding-ada-002"

        GITHUB_BASE_URL: str = "https://api.github.com"
        GITHUB_TOKEN: str = ""
        GITHUB_USER_NAME: str = ""
        GITHUB_REPO_NAME: str = ""

        SYSTEM_PROMPT: str = "You are a smart assistant that read from github repository, retrieves their information, analyzes them, and assists users with Q&A over extracted content."
        
    def __init__(self):

        self.name = "Chat with GitHub Repository"
        self.check = 0

        self.valves = self.Valves(
            OPENAI_API_BASE_URL = os.getenv("OPENAI_API_BASE_URL", ""),
            OPENAI_API_MODEL = os.getenv("OPENAI_API_MODEL", ""),
            OPENAI_API_TEMPERATURE = float(os.getenv("OPENAI_API_TEMPERATURE"), ""),
            OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", ""),
            OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", ""),

            GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", ""),
            GITHUB_BASE_URL = os.getenv("GITHUB_BASE_URL", ""),
            GITHUB_USER_NAME = os.getenv("GITHUB_USER_NAME", ""),
            GITHUB_REPO_NAME = os.getenv("GITHUB_REPO_NAME", ""),

            SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", ""),
        )

        self.tools = [search_repository]
    
    def set_github_repo(self):
        """Loads GitHub repository data and creates an index."""
        try:
            global index, documents
            
            repository_url = f"{self.valves.GITHUB_BASE_URL}/repos/{self.valves.GITHUB_USER_NAME}/{self.valves.GITHUB_REPO_NAME}"
            headers = { 'Authorization': f"Bearer {self.valves.GITHUB_TOKEN}" }
            
            embed_model = OpenAIEmbedding(model=self.valves.OPENAI_EMBED_MODEL, api_key=self.valves.OPENAI_API_KEY)
            
            try:
                response = requests.get(repository_url, headers=headers)
                if response.status_code == 200:
                    repositories = response.json()
                        
                    file_paths = self.get_all_files(repository_url, headers)

                    files_data = []
                    for file_path in file_paths:
                        content = self.get_file_content(repository_url, file_path, headers)
                        if content:
                            files_data.append({"path": file_path, "content": content})

                    documents = [
                        Document(
                            text=f"Github URL: {repository_url}",
                            metadata={"type": "repo_info", "key": "Github URL"}
                        ),
                        Document(
                            text=f"Project name: {repositories.get('name', 'Unknown')}",
                            metadata={"type": "repo_info", "key": "Project Name"}
                        ),
                        Document(
                            text=f"Project owner: {repositories.get('owner', {}).get('login', 'Unknown')}",
                            metadata={"type": "repo_info", "key": "Project Owner"}
                        ),
                        Document(
                            text=f"List users with access: {self.get_collaborators(repositories.get('collaborators_url', '').split('{')[0])}",
                            metadata={"type": "repo_info", "key": "Users with Access"}
                        ),
                        Document(
                            text=f"Programming languages used: {self.get_languages(repositories.get('languages_url', ''))}",
                            metadata={"type": "repo_info", "key": "Languages Used"}
                        ),
                        Document(
                            text=f"Security/visibility level: {repositories.get('visibility', 'Unknown')}",
                            metadata={"type": "repo_info", "key": "Visibility"}
                        ),
                        Document(
                            text=f"Summary: {repositories.get('description', 'No description')}",
                            metadata={"type": "repo_info", "key": "Summary"}
                        ),
                        Document(
                            text=f"Last maintained: {repositories.get('pushed_at', 'Unknown')}",
                            metadata={"type": "repo_info", "key": "Last Maintained"}
                        ),
                        Document(
                            text=f"Last release: {repositories.get('default_branch', 'Unknown')}",
                            metadata={"type": "repo_info", "key": "Last Release"}
                        ),
                        Document(
                            text=f"Open issues: {self.get_open_issues(repository_url, headers)}",
                            metadata={"type": "repo_info", "key": "Open Issues"}
                        )
                    ]

                    for file in files_data:
                        documents.append(Document(
                        text=f"File: {file['path']}\nContent:\n{file['content']}",
                        metadata={"type": "file", "file_path": file["path"]}
                    ))

                else:
                    print(f"Failed to retrieve repositories. Status code: {response.status_code}")
                
            except Exception as e:
                print(f"Error: {e}")

            try:
                index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
            except Exception as e:
                print(f"Error while indexing: {str(e)}")

            print("GitHub repository indexed successfully!")
                
        except Exception as e:
            print(f"Error in on_startup: {str(e)}")

    def get_collaborators(self, collaborators_url):
        response = requests.get(collaborators_url)
        if response.status_code == 200:
            return [collaborator["login"] for collaborator in response.json()]
        else:
            return []

    def get_languages(self, languages_url):
        response = requests.get(languages_url)
        if response.status_code == 200:
            return list(response.json().keys())
        else:
            return []

    def get_open_issues(self, url, headers):
        url = f"{url}/issues?state=open"
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error: {response.status_code}")
            return []

    def get_all_files(self, url, headers):
        url = f"{url}/git/trees/main?recursive=1"
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            return [item["path"] for item in data.get("tree", []) if item["type"] == "blob"]
        else:
            print(f"Error: {response.status_code}")
            return []

    def get_file_content(self, url, file_path, headers):
        url = f"{url}/contents/{file_path}"
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            file_data = response.json()
            content = file_data.get("content", "")
            encoding = file_data.get("encoding", "")

            if encoding == "base64":
                decoded_content = base64.b64decode(content)
                try:
                    return decoded_content.decode("utf-8")  # Try decoding as UTF-8 text
                except UnicodeDecodeError:
                    return "Binary Files"  # Return raw bytes for binary files
            else:
                print(f"Unknown encoding for {file_path}: {encoding}")
                return None
        else:
            print(f"Error fetching {file_path}: {response.status_code}")
            return None

    def get_openai_models(self):
        if self.valves.OPENAI_API_KEY:
            try:
                headers = {
                    "Authorization": f"Bearer {self.valves.OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                }
                response = requests.get(
                    f"{self.valves.OPENAI_API_BASE_URL}/models", headers=headers
                )
                models = response.json()
                return [
                    {"id": model["id"], "name": model.get("name", model["id"])}
                    for model in models["data"] if "gpt" in model["id"]
                ]
            except Exception as e:
                print(f"Error: {e}")
                return [{"id": "error", "name": "Could not fetch models from OpenAI."}]
        else:
            return []


    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict):
        """Processes user messages and interacts with the LLM."""
        try:
            if self.check == 0:
                self.pipelines = self.get_openai_models()
                self.set_github_repo()
                self.check = 1

            model = ChatOpenAI(
                api_key=self.valves.OPENAI_API_KEY,
                model=self.valves.OPENAI_API_MODEL,
                temperature=self.valves.OPENAI_API_TEMPERATURE
            )
            
            tools: Sequence[BaseTool] = self.tools

            prompt = ChatPromptTemplate.from_messages([
                ("system", self.valves.SYSTEM_PROMPT),
                MessagesPlaceholder("chat_history"),
                ("user", "{input}"),
                MessagesPlaceholder("agent_scratchpad")
            ])
            agent = create_tool_calling_agent(model, tools, prompt)
            agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
            response = agent_executor.invoke({"input": user_message, "chat_history": messages})
            return response["output"]
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            raise
