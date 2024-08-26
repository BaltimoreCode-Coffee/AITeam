# AITeam
Monorepo for the Baltimore Code and Coffee AI Team


## Contributing to the Project

Thank you for your interest in contributing to the Baltimore Code and Coffee website! Below are instructions to help you get started with testing your changes locally, creating pull requests, and ensuring your contributions follow our guidelines.

### How To Get the files, Test Locally, and create a Pull Request (Contribute!)

1. **Download Github Desktop**: This tool makes using Git easy and fun! Command line instructions are also included in this guide    
   **Windows:**  
https://desktop.github.com/download/  
   **Ubuntu**  
```bash
sudo wget https://github.com/shiftkey/desktop/releases/download/release-3.1.1-linux1/GitHubDesktop-linux-3.1.1-linux1.deb
### Uncomment below line if you have not installed gdebi-core before
# sudo apt-get install gdebi-core 
sudo gdebi GitHubDesktop-linux-3.1.1-linux1.deb
```

2. **Fork the repository**:
  At the top of this page, click "Fork", and make your own copy of this repository
2. **Clone the repository**:  
   **GitHub Desktop:** 
   - Open GitHub Desktop and go to `File` -> `Clone Repository`.
   - Select the URL tab and paste the repository link: `https://github.com/YOUR_ACCOUNT/BaltimoreCode-Coffee.github.io.git`
   - Make sure to clone the one from your own account, so that you have write permission!
   - Choose the local path where you want to clone the repository.
   - Click `Clone`.
   
   or if you prefer the command line:
   ```bash
   git clone https://github.com/YOUR_ACCOUNT/BaltimoreCode-Coffee.github.io.git
   ```

3. **Open VSCode** to the BaltimoreCode-Coffee.github.io folder
4. **Open a Terminal in VSCode**
5. **Install http-server**:
   ```bash
   npm install -g http-server
   ```
   
6. **Serve the project** without caching:
     ```bash
     http-server -c-1
     ```
   
7. **Access the site** in your browser:
   - Open your browser and navigate to `http://localhost:8080`. You should see the website as it would appear with your changes applied.

8. **Make Changes** then refresh your browser page
    Once you've made and tested your changes, you can submit them for review by creating a pull request:

9. **Commit your changes**:
   - In GitHub Desktop:
     - Write a summary of your changes in the `Summary` field.
     - Optionally, add a description.
     - Click `Commit to <branch-name>`.
   
   or via command line:
   ```bash
   git add .
   git commit -m "Description of your changes"
   ```
   
8. **Push your branch** to the remote repository:
   - In GitHub Desktop:
     - Click `Push origin` in the toolbar.
   
   or via command line:
   ```bash
   git push origin <branch-name>
   ```
   
9. **Create a pull request**:
   - In GitHub Desktop:
     - Click `Branch` -> `Create Pull Request` to open the GitHub page.
   
   or via GitHub web:
   - Go to the repository on GitHub.
   - Click on the "Compare & pull request" button next to your branch.
   - Add a clear title and description for your pull request, explaining what changes you made and why.
   - Click "Create pull request".
