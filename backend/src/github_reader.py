import json
import os
import requests
from enum import Enum
from typing import List, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class FileType(Enum):
    FILE = "file"
    DIRECTORY = "dir"

class FileNode:
    def __init__(self, path: str = '', typ: FileType = FileType.DIRECTORY, size: Optional[int] = None):
        self.path = path
        self.type = typ
        self.size = size
        self.children = list()

    def add_child(self, child):
        self.children.append(child)

    def __str__(self):
        return ', '.join(['%s: %s' % item for item in self.__dict__.items()])

def json_serializer(obj):
    if isinstance(obj, FileNode):
        return obj.__dict__
    if isinstance(obj, FileType):
        return obj.value
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

def join_path(pa, pb):
    return pa + ('/' if len(pa) else '') + pb

def get_default_branch(owner, repo):
    """
    Get the default branch of the repository
    """
    url = f'https://api.github.com/repos/{owner}/{repo}'
    headers = {'User-Agent': 'request'}
    
    # Add GitHub token for authentication if available
    github_token = os.getenv('GITHUB_TOKEN')
    if github_token:
        headers['Authorization'] = f'token {github_token}'
    
    resp = requests.get(url, headers=headers)

    if resp.status_code == 200:
        repo_info = resp.json()
        return repo_info['default_branch']
    else:
        raise Exception(f"Error: Unable to fetch repository info. Status code: {resp.status_code}")

def get_repo_tree(owner, repo, sha):
    """
    Get the directory tree of the specified branch
    """
    url = f'https://api.github.com/repos/{owner}/{repo}/git/trees/{sha}?recursive=1'
    headers = {'User-Agent': 'request'}
    
    # Add GitHub token for authentication if available
    github_token = os.getenv('GITHUB_TOKEN')
    if github_token:
        headers['Authorization'] = f'token {github_token}'
    
    resp = requests.get(url, headers=headers)

    if resp.status_code == 200:
        return resp.json()
    else:
        raise Exception(f"Error: Unable to fetch repository tree. Status code: {resp.status_code}")

def get_base_url(path: str) -> str:
    path = path.strip('/')
    i = len(path) - 1
    while i >= 0:
        if path[i] == '/':
            return path[:i]
        i -= 1
    return ''

def get_tree_by_path(owner: str, repo: str, path: str) -> List[FileNode]:
    """
    Get the directory tree under the specified path
    """
    default_branch = get_default_branch(owner, repo)
    tree = get_repo_tree(owner, repo, default_branch)

    nodes = {path: FileNode(path)}
    for child in tree.get('tree'):
        cpath: str = child.get('path', '')
        ctype = FileType.DIRECTORY if child.get('type') == 'tree' else FileType.FILE
        if len(cpath) > len(path) and cpath.startswith(path):
            node = FileNode(cpath, ctype, child.get('size'))
            nodes[cpath] = node
            nodes[get_base_url(cpath)].add_child(node)
    return nodes[path].children

def get_files_by_path(owner: str, repo: str, path: str) -> List[FileNode]:
    """
    Get all files under the specified path
    """
    get_repository_content_url = f'https://api.github.com/repos/{owner}/{repo}/contents/{path}'
    headers = {'User-Agent': 'request'}
    
    # Add GitHub token for authentication if available
    github_token = os.getenv('GITHUB_TOKEN')
    if github_token:
        headers['Authorization'] = f'token {github_token}'
    
    resp = requests.get(get_repository_content_url, headers=headers)

    if resp.status_code == 200:
        children = list()
        for item in resp.json():
            child = FileNode(join_path(path, item.get('name')), FileType.FILE if item.get('type') == 'file' else FileType.DIRECTORY, item.get('size'))
            children.append(child)
        return children
    else:
        raise Exception(f'Error: Unable to fetch repository tree. Status code: {resp.status_code}')

def read_project_structure(owner: str, repo: str, path: str, recursion: bool) -> dict:
    try:
        path = path.strip('/')
        if recursion:
            children = get_tree_by_path(owner, repo, path)
        else:
            children = get_files_by_path(owner, repo, path)
        return {"children": json.dumps(children, default=json_serializer), "status": True, "status_msg": "ok"}
    except Exception as e:
        return {"status": False, "status_msg": str(e)}

def find_markdown_files(owner: str, repo: str, branch: str = None, path: str = "", url_base: str = None) -> dict:
    """
    Find all markdown files in a repository and generate raw GitHub URLs
    
    Args:
        owner: Repository owner
        repo: Repository name
        branch: Branch to use (if None, default branch will be used)
        path: Path within the repository to start searching from
        url_base: Base URL for raw content (if None, will be constructed)
        
    Returns:
        Dictionary with "urls" list containing raw githubusercontent URLs to markdown files
    """
    try:
        # Get default branch if not specified
        if branch is None:
            branch = get_default_branch(owner, repo)
            
        # Get file tree
        tree = get_repo_tree(owner, repo, branch)
        
        # Base URL for raw content
        if url_base is None:
            url_base = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/"
        
        # List to store markdown file paths
        md_paths = []
        
        # Find all markdown files in the tree
        for item in tree.get('tree', []):
            item_path = item.get('path', '')
            item_type = item.get('type')
            
            # Check if it's a file with .md extension and in the right path
            if (item_type == 'blob' and 
                item_path.endswith('.md') and 
                (not path or item_path.startswith(path))):
                md_paths.append(url_base + item_path)
        
        return {"urls": md_paths, "status": True, "status_msg": "ok"}
    except Exception as e:
        return {"status": False, "status_msg": str(e)}

def find_markdown_files_from_url(github_url: str) -> dict:
    """
    Find all markdown files from a GitHub repository URL
    
    Args:
        github_url: GitHub URL in format https://github.com/{owner}/{repo}/tree/{branch}/{path}
        
    Returns:
        Dictionary with "urls" list containing raw githubusercontent URLs to markdown files
    """
    try:
        # Parse GitHub URL
        parts = github_url.replace("https://github.com/", "").split("/")
        
        if len(parts) < 2:
            return {"status": False, "status_msg": "Invalid GitHub URL format"}
            
        owner = parts[0]
        repo = parts[1]
        
        # Handle branch and path
        branch = None
        path = ""
        
        if len(parts) > 3 and parts[2] == "tree":
            branch = parts[3]
            path = "/".join(parts[4:]) if len(parts) > 4 else ""
            
        # Create base URL for raw content (similar to JS implementation)
        url_base = github_url.replace('github.com', 'raw.githubusercontent.com').replace('/tree/', '/') 
        if '/tree/' in github_url:
            url_base = url_base[:url_base.rindex('/')+1]
        elif not url_base.endswith('/'):
            url_base += '/'
            
        return find_markdown_files(owner, repo, branch, path, url_base)
    except Exception as e:
        return {"status": False, "status_msg": str(e)}

def find_markdown_files_in_repo(owner: str, repo: str, path: str = "") -> dict:
    """
    Find all markdown files in a repository and generate raw GitHub URLs
    
    Args:
        owner: Repository owner
        repo: Repository name
        path: Path within the repository to start searching from
        
    Returns:
        Dictionary with "urls" list containing raw githubusercontent URLs to markdown files
    """
    try:
        return find_markdown_files(owner, repo, path=path)
    except Exception as e:
        return {"status": False, "status_msg": str(e)}

def extract_markdown_urls_from_tree(file_tree_json: str, github_url: str) -> dict:
    """
    Extract all markdown file URLs from a file tree JSON
    
    Args:
        file_tree_json: JSON string representing the file tree
        github_url: GitHub URL for the repository
        
    Returns:
        Dictionary with "urls" list containing raw githubusercontent URLs to markdown files
    """
    try:
        # Parse the JSON string (handle escape characters like in JS)
        file_tree_json = file_tree_json.replace('\\', '')
        file_tree = json.loads(file_tree_json)
        
        # Create the URL base similar to the JS implementation
        url_base = github_url.replace('github.com', 'raw.githubusercontent.com')
        url_base = url_base.replace('/tree/', '/refs/heads/')
        
        # Make sure url_base ends with a slash
        if '/' in url_base:
            url_base = url_base[:url_base.rindex('/')+1]
        elif not url_base.endswith('/'):
            url_base += '/'
            
        # List to store markdown file paths
        md_paths = []
        
        # Recursive function to find markdown files
        def find_md_files_recursive(nodes):
            if not isinstance(nodes, list):
                return
                
            for node in nodes:
                # Check if it's a .md file
                if (isinstance(node, dict) and 
                    node.get('type') == 'file' and 
                    isinstance(node.get('path'), str) and 
                    node.get('path').endswith('.md')):
                    md_paths.append(url_base + node.get('path'))
                    
                # If it's a directory with children, recurse
                if (isinstance(node, dict) and 
                    node.get('type') == 'dir' and 
                    isinstance(node.get('children'), list) and 
                    len(node.get('children')) > 0):
                    find_md_files_recursive(node.get('children'))
        
        # Start recursion
        find_md_files_recursive(file_tree)
        
        return {"urls": md_paths, "status": True, "status_msg": "ok"}
    except Exception as e:
        return {"status": False, "status_msg": str(e)}

def get_markdown_urls_from_github(github_url: str) -> list:
    """
    Get all markdown file URLs from a GitHub repository URL in a single function call
    
    Args:
        github_url: GitHub URL in format https://github.com/{owner}/{repo}/tree/{branch}/{path}
        
    Returns:
        List of raw GitHub URLs to markdown files
    """
    try:
        # Parse GitHub URL to get owner, repo, branch, path
        parts = github_url.replace("https://github.com/", "").split("/")
        
        if len(parts) < 2:
            print("Error: Invalid GitHub URL format")
            return []
            
        owner = parts[0]
        repo = parts[1]
        
        # Handle branch and path
        branch = None
        path = ""
        
        if len(parts) > 3 and parts[2] == "tree":
            branch = parts[3]
            path = "/".join(parts[4:]) if len(parts) > 4 else ""
        
        # Get project structure
        structure = read_project_structure(owner, repo, path, True)
        
        if not structure.get("status", False):
            print(f"Error: {structure.get('status_msg', 'Unknown error')}")
            return []
            
        # Extract markdown URLs
        md_result = extract_markdown_urls_from_tree(structure['children'], github_url)
        
        return md_result.get("urls", [])
    except Exception as e:
        print(f"Error: {str(e)}")
        return []

# Example usage
if __name__ == "__main__":
    urls = get_markdown_urls_from_github('https://github.com/jax-ml/jax/tree/main/docs')
    print(urls)