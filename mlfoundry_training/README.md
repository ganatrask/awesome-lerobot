This script automates your Lerobot setup process on Foundry. To use it:

Save the script (e.g., as lerobot_pi0_setup.sh)
Make it executable: chmod +x lerobot_pi0_setup.sh
Run it: ./lerobot_pi0_setup.sh or you can add this script while requesting the instance so it begins at the  start of instance 

The script will:

Clean up any existing installations
Clone the LERobot repository
Check out the specific commit you used (b536f47)
Create a fresh virtual environment
Install all dependencies with the correct versions
Provide instructions for the final authentication steps

I left the `wandb login` and `huggingface-cli login` commands as manual steps since they require interactive authentication. You'll need to run those after the script completes.
