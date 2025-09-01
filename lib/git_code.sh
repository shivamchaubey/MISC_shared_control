# 0) Create your repo locally and on GitHub
git init
git add .
git commit -m "init"

# 1) Add the external repo as a submodule at your desired path
git submodule add -b main https://github.com/OTHER_USER/OTHER_REPO.git lib/matlab/OTHER_REPO

# 2) Commit the .gitmodules file + submodule pointer
git commit -m "Add submodule OTHER_REPO under lib/matlab/OTHER_REPO"

# 3) Push your repo
git remote add origin https://github.com/YOU/YOUR_REPO.git
git push -u origin main