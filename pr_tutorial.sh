# <fork_name> - is a name of the person who created the PR
# <PR_name> - is a name of the PR
# Example: Jonas-Verhellen:eccentricity-lifting --> 
# --> <fork_name> = jonas-verhellen <PR_name> = eccentricity-lifting

# First it is required to fetch the PR
git fetch <fork_name> <PR_name>

# Check that the PR is fetched
git branch -r

# Checkout to the PR 
git checkout <fork_name>/<PR_name>

# Push to the PR
git push <fork_name> <PR_name>