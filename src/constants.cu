#include "constants.h"

std::string toAbsFilePath(const std::string& relpath){

    // Get Current Path
    fs::path currentPath = fs::current_path();

    // Move 2 folder up to Workspace Level
    currentPath = currentPath.parent_path();

    // Append the relative path
    currentPath.append(relpath);

    std::string filepath = currentPath.string();
    // Trim any potential whitespace or newline characters
    filepath.erase(filepath.find_last_not_of(" \n\r\t") + 1);
    
    return filepath;


}


Config::Config(const std::string& filename) {
        std::ifstream configFile(toAbsFilePath(filename));
        if (configFile.is_open()) {
            std::string line;
            while (std::getline(configFile, line)) {
                // Skip comments and empty lines
                if (line.empty() || line[0] == ';' || line[0] == '#') {
                    continue;
                }

                // Skip section headers
                if (line[0] == '[') {
                    continue;
                }

                std::istringstream is_line(line);
                std::string key;
                if (std::getline(is_line, key, '=')) {
                    std::string value;
                    if (std::getline(is_line, value)) {
                        // Remove leading and trailing whitespaces from key and value
                        key = trim(key);
                        value = trim(value);
                        
                        if (key == "DX") {
                            DX = std::stod(value);
                        } else if (key == "DY") {
                            DY = std::stod(value);
                        } else if (key == "mu") {
                            mu = std::stod(value);
                        } else if (key == "dt") {
                            dt = std::stod(value);
                        } else if (key == "T") {
                            T = std::stod(value);
                        }
                    }
                }
            }
            configFile.close();
        } else {
            std::cerr << "Unable to open config file" << std::endl;
        }
}


std::string Config::trim(const std::string& str) {
        size_t first = str.find_first_not_of(' ');
        if (first == std::string::npos) {
            return "";
        }
        size_t last = str.find_last_not_of(' ');
        return str.substr(first, last - first + 1);
}