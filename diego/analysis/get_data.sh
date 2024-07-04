#!/bin/bash

# Function to rsync file from remote server to local machine
rsync_remote_to_local() {
    remote_host="$1"
    remote_base_path="$2"
    local_base_path="$3"
    folder_list=("$@")

    for folder in "${folder_list[@]}"; do
        echo $folder
        echo $folder_list
        remote_folder="$remote_base_path/$folder"
        local_folder="$local_base_path/$folder"
        
        echo $remote_base_path
        echo $folder
        subfolder_list=$(ssh "$remote_host" "find $remote_folder -mindepth 1 -maxdepth 1 -type d -printf '%f\n'")
        
        for subfolder in $subfolder_list; do
            remote_subfolder="$remote_folder/$subfolder"
            local_subfolder="$local_folder/$subfolder"
            

            # Make sure the local subfolder exists
            mkdir -p "$local_subfolder"

            # Rsync files from the remote subfolder to the local subfolder
            rsync -avz -e ssh "$remote_host:$remote_subfolder/statistics_target.pkl" "$local_subfolder/"
        done
    done
}

# Example usage

remote_host="ddoimo@195.14.102.215"
remote_base_path="/u/area/ddoimo/ddoimo/open/geometric_lens/repo/results/mmlu"
local_base_path="./results/mmlu"
model_list=("llama-2-7b" "llama-2-7b-chat" "llama-2-13b" "llama-2-13b-chat" "llama-2-70b" "llama-2-70b-chat" "llama-3-8b" "llama-3-8b-chat" "llama-3-70b" "llama-3-70b-chat")



for folder in "${model_list[@]}"; do
    remote_folder="$remote_base_path/$folder"
    local_folder="$local_base_path/$folder"
    
    subfolder_list=$(ssh "$remote_host" "find $remote_folder -mindepth 1 -maxdepth 1 -type d -printf '%f\n'")
    
    for subfolder in $subfolder_list; do
        remote_subfolder="$remote_folder/$subfolder"


        local_subfolder="$local_folder/$subfolder"
        echo $local_subfolder
        

        # Make sure the local subfolder exists
        mkdir -p "$local_subfolder"

        # Rsync files from the remote subfolder to the local subfolder
        rsync -avz -e ssh "$remote_host:$remote_subfolder/statistics_target.pkl" "$local_subfolder/"
    done
done

#rsync_remote_to_local "$remote_host" "$remote_base_path" "$local_base_path" "${model_list[@]}"