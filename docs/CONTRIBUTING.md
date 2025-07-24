## 🤝 Contributing

We welcome contributions! Follow these steps to get started:

1. **Fork the Repository**

   Click the **"Fork"** button at the top-right of the [MageML GitHub page](https://github.com/trentonom0r3/MageML).

2. **Clone Your Fork Locally**

   ```bash
   git clone https://github.com/trentonom0r3/MageML.git
   cd MageML
   git submodule update --init --recursive
   ```

3. **Set Up the Project**

   - Open `CMakePresets.json` and adjust paths as needed (e.g., for dependencies or build directories).  
   - Open the repo root folder in **Visual Studio** (or your preferred CMake-aware IDE).

4. **Make Your Changes**

   - Work in a new feature branch:  
     ```bash
     git checkout -b feature/my-new-feature
     ```
   - Keep commits focused and descriptive.
   - Comment code where necessary and follow existing conventions.

5. **Submit a Pull Request**

   - Push your changes to your fork:  
     ```bash
     git push origin feature/my-new-feature
     ```
   - Go to your fork on GitHub and click **"Compare & pull request"**.
   - Include a clear description and comments explaining your changes.

> 📝 We appreciate detailed PRs with context and reasoning. Screenshots, test results, and code comments are always welcome!
