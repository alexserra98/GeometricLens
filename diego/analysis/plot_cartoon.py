import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
import matplotlib.patches as patches

# Number of Gaussians and points per Gaussian
rng = np.random.default_rng(seed=11)

path = "/home/diego/Documents/area_science/ricerca/open/geometric_lens/repo/diego/analysis/plots/final"


# ***********************************************


# points_per_gaussian = 100


# # Parameters for the Gaussians
# std = 1.3e-3
# covariance = [[std, 0], [0, std]]  # Same covariance matrix (variance)


# # Define custom colors for each cluster
# custom_colors = {
#     "math": "C3",
#     "physics": "C1",
#     "chemistry": "C4",  # gold",
#     "philosopy": "C0",
#     "history": "C2",
#     "statistics": "C5",
#     "machine\nlearning": "C7",
#     "biology": "C8",
#     "anatomy": "C6",
# }


# #########################

# # 0-SHOT

# #########################

# # Parameters for the Gaussians
# std = 2.3e-3
# covariance = [[std, 0], [0, std]]  # Same covariance matrix (variance)


# subjects = {
#     "math": (0.25, 0.45),
#     "physics": (0.4, 0.4),
#     "chemistry": (0.4, 0.3),
#     "philosopy": (0.75, 0.6),
#     "history": (0.8, 0.7),
#     "statistics": (0.2, 0.8),
#     "machine\nlearning": (0.3, 0.85),
#     "biology": (0.75, 0.2),
#     "anatomy": (0.85, 0.2),
# }

# # Generate points for each Gaussian and create a DataFrame
# data = []
# for subject, mean in subjects.items():
#     points = rng.multivariate_normal(mean, covariance, points_per_gaussian)
#     df = pd.DataFrame(points, columns=["x", "y"])
#     df["label"] = subject
#     data.append(df)
# # Combine all data into a single DataFrame
# data = pd.concat(data)

# position = {
#     "math": (0.15, 0.15),
#     "physics": (0.15, 0.1),
#     "chemistry": (0.15, 0.05),
#     "philosopy": (0.85, 0.92),
#     "history": (0.85, 0.87),
#     "statistics": (0.52, 0.82),
#     "machine\nlearning": (0.52, 0.92),
#     "biology": (0.88, 0.4),
#     "anatomy": (0.88, 0.35),
# }


# fig = plt.figure(figsize=(12, 3))


# ##############################################

# gs = GridSpec(1, 1)
# ax = fig.add_subplot(gs[0])
# sns.scatterplot(
#     data=data, x="x", y="y", hue="label", palette=custom_colors, s=30, alpha=0.9
# )

# plt.legend([], [], frameon=False)  # Remove the legend
# plt.xticks([])  # Remove x-axis ticks
# plt.yticks([])  # Remove y-axis ticks
# plt.xlabel("")  # Remove x-axis label
# plt.ylabel("")  # Remove y-axis label

# ax.set_xlim(0, 1)
# ax.set_ylim(0, 1)


# # Add custom labels near each cluster
# for label, mean in subjects.items():
#     pos = position[label]
#     plt.text(
#         pos[0],
#         pos[1],
#         label,
#         fontsize=9,
#         color="black",
#         weight="bold",
#         va="center",
#         ha="center",
#     )


# ax.set_title("0-shot", weight="bold", fontsize=11)
# ax.set_ylabel("early layers", weight="bold", fontsize=13)
# gs.tight_layout(fig, rect=[0.0, 0.0, 0.33, 1])

# #################

# # FEW_SHOT

# #################


# # Parameters for the Gaussians
# std = 1.6e-3
# covariance = [[std, 0], [0, std]]  # Same covariance matrix (variance)

# subjects_fs = {
#     "math": (0.28, 0.47),
#     "physics": (0.4, 0.38),
#     "chemistry": (0.35, 0.2),
#     "philosopy": (0.65, 0.6),
#     "history": (0.8, 0.7),
#     "statistics": (0.15, 0.7),
#     "machine\nlearning": (0.25, 0.88),
#     "biology": (0.7, 0.2),
#     "anatomy": (0.85, 0.15),
# }


# # Generate points for each Gaussian and create a DataFrame
# data_fs = []
# for subject, mean in subjects_fs.items():
#     points = rng.multivariate_normal(mean, covariance, points_per_gaussian)
#     df = pd.DataFrame(points, columns=["x", "y"])
#     df["label"] = subject
#     data_fs.append(df)
# # Combine all data into a single DataFrame
# data_fs = pd.concat(data_fs)

# position_fs = {
#     "math": (-0.15, -0.1),
#     "physics": (0.18, 0.0),
#     "chemistry": (-0.15, -0.08),
#     "philosopy": (0.2, -0.08),
#     "history": (0.1, 0.15),
#     "statistics": (0.2, 0),
#     "machine\nlearning": (0.2, 0),
#     "biology": (-0.03, -0.12),
#     "anatomy": (0.05, 0.12),
# }


# gs = GridSpec(1, 1)
# ax = fig.add_subplot(gs[0])
# sns.scatterplot(
#     data=data_fs, x="x", y="y", hue="label", palette=custom_colors, s=30, alpha=0.9
# )

# plt.legend([], [], frameon=False)  # Remove the legend
# plt.xticks([])  # Remove x-axis ticks
# plt.yticks([])  # Remove y-axis ticks
# plt.xlabel("")  # Remove x-axis label
# plt.ylabel("")  # Remove y-axis label

# ax.set_xlim(0, 1)
# ax.set_ylim(0, 1)

# # Add custom labels near each cluster
# for label, mean in subjects_fs.items():
#     pos = position_fs[label]
#     plt.text(
#         mean[0] + pos[0],
#         mean[1] + pos[1],
#         label,
#         fontsize=9,
#         color="black",
#         weight="bold",
#         va="center",
#         ha="center",
#     )
# ax.set_title("few-shot", weight="bold", fontsize=11)

# gs.tight_layout(fig, rect=[0.38, 0.0, 0.69, 1])


# ########################

# # FINETUNED

# ########################

# # Parameters for the Gaussians
# std = 2e-3
# covariance = [[std, 0], [0, std]]  # Same covariance matrix (variance)


# subjects_ft = {
#     "math": (0.25, 0.45),
#     "physics": (0.4, 0.4),
#     "chemistry": (0.4, 0.3),
#     "philosopy": (0.75, 0.6),
#     "history": (0.8, 0.7),
#     "statistics": (0.2, 0.8),
#     "machine\nlearning": (0.3, 0.85),
#     "biology": (0.75, 0.2),
#     "anatomy": (0.85, 0.2),
# }


# # Generate points for each Gaussian and create a DataFrame
# data_ft = []
# for subject, mean in subjects_ft.items():
#     points = rng.multivariate_normal(mean, covariance, points_per_gaussian)
#     df = pd.DataFrame(points, columns=["x", "y"])
#     df["label"] = subject
#     data_ft.append(df)
# # Combine all data into a single DataFrame
# data_ft = pd.concat(data_ft)

# position_ft = {
#     "math": (0.15, 0.15),
#     "physics": (0.15, 0.1),
#     "chemistry": (0.15, 0.05),
#     "philosopy": (0.85, 0.92),
#     "history": (0.85, 0.87),
#     "statistics": (0.52, 0.82),
#     "machine\nlearning": (0.52, 0.92),
#     "biology": (0.88, 0.4),
#     "anatomy": (0.88, 0.35),
# }

# gs = GridSpec(1, 1)
# ax = fig.add_subplot(gs[0])
# sns.scatterplot(
#     data=data_ft, x="x", y="y", hue="label", palette=custom_colors, s=30, alpha=0.9
# )

# plt.legend([], [], frameon=False)  # Remove the legend
# plt.xticks([])  # Remove x-axis ticks
# plt.yticks([])  # Remove y-axis ticks
# plt.xlabel("")  # Remove x-axis label
# plt.ylabel("")  # Remove y-axis label

# ax.set_xlim(0, 1)
# ax.set_ylim(0, 1)


# # Add custom labels near each cluster
# for label, mean in subjects_ft.items():
#     pos = position_ft[label]
#     plt.text(
#         pos[0],
#         pos[1],
#         label,
#         fontsize=9,
#         color="black",
#         weight="bold",
#         va="center",
#         ha="center",
#     )
# ax.set_title("finetuned", weight="bold", fontsize=11)

# gs.tight_layout(fig, rect=[0.69, 0, 1, 1])
# plt.savefig(f"{path}/cartoon_subjects.png", dpi=200)


##########################
##########################

# SUBJECTS LESS

#########################
#########################


points_per_gaussian = 100
background_color = "peru"  # "antiquewhite"# "papayawhip"
bg_alpha = 0.1
dp_alpha = 1
title_size = 14

# # Parameters for the Gaussians
# std = 1.3e-3
# covariance = [[std, 0], [0, std]]  # Same covariance matrix (variance)


# Define custom colors for each cluster
custom_colors = {
    "math": "C3",
    "physics": "C1",
    "chemistry": "C4",  # gold",
    "philosopy": "C0",
    "history": "C2",
    "statistics": "C5",
    "machine\nlearning": "C7",
}


#########################

# 0-SHOT

#########################

# Parameters for the Gaussians
std = 1.5e-3
covariance = [[std, 0], [0, std]]  # Same covariance matrix (variance)


subjects = {
    "math": (0.45, 0.3),
    "physics": (0.55, 0.3),
    "chemistry": (0.65, 0.2),
    "philosopy": (0.8, 0.65),
    "history": (0.85, 0.75),
    "statistics": (0.25, 0.65),
    "machine\nlearning": (0.2, 0.75),
}

# Generate points for each Gaussian and create a DataFrame
data = []
for subject, mean in subjects.items():
    points = rng.multivariate_normal(mean, covariance, points_per_gaussian)
    df = pd.DataFrame(points, columns=["x", "y"])
    df["label"] = subject
    data.append(df)
# Combine all data into a single DataFrame
data = pd.concat(data)

position = {
    "math": (0.85, 0.2),
    "physics": (0.85, 0.15),
    "chemistry": (0.85, 0.1),
    "philosopy": (0.85, 0.95),
    "history": (0.85, 0.9),
    "statistics": (0.4, 0.8),
    "machine\nlearning": (0.4, 0.9),
}


fig = plt.figure(figsize=(12, 3))


##############################################

gs = GridSpec(1, 1)
ax = fig.add_subplot(gs[0])
sns.scatterplot(
    data=data, x="x", y="y", hue="label", palette=custom_colors, s=30, alpha=dp_alpha
)

plt.legend([], [], frameon=False)  # Remove the legend
plt.xticks([])  # Remove x-axis ticks
plt.yticks([])  # Remove y-axis ticks
plt.xlabel("")  # Remove x-axis label
plt.ylabel("")  # Remove y-axis label

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)


# Add custom labels near each cluster
for label, mean in subjects.items():
    pos = position[label]
    plt.text(
        pos[0],
        pos[1],
        label,
        fontsize=9,
        color="black",
        weight="bold",
        va="center",
        ha="center",
    )


ax.set_title("zero shot", weight="bold", fontsize=title_size)
ax.set_ylabel("early layers", weight="bold", fontsize=13)


ax.plot(0.34, 0.45, marker="^", markersize=9, color="black")
ax.plot(0.3, 0.45, marker="^", markersize=9, color="black")

ax.plot(0.68, 0.5, marker="^", markersize=9, color="black")
ax.plot(0.64, 0.5, marker="^", markersize=9, color="black")


# legend
ax.plot(0.06, 0.07, marker="^", markersize=7, color="black")
ax.plot(0.08, 0.07, marker="^", markersize=7, color="black")
plt.text(0.11, 0.06, "saddle point", fontsize=11, color="black", va="center")

rect = patches.Rectangle(
    (0, 0),
    1,
    1,
    transform=ax.transAxes,
    color=background_color,
    alpha=bg_alpha,
    zorder=0,
)
ax.add_patch(rect)


gs.tight_layout(fig, rect=[0.0, 0.0, 0.33, 1])

#################

# FEW_SHOT

#################


# Parameters for the Gaussians
std = 0.6e-3
covariance = [[std, 0], [0, std]]  # Same covariance matrix (variance)

subjects_fs = {
    "math": (0.4, 0.3),
    "physics": (0.55, 0.35),
    "chemistry": (0.75, 0.2),
    "philosopy": (0.68, 0.65),
    "history": (0.8, 0.8),
    "statistics": (0.2, 0.6),
    "machine\nlearning": (0.18, 0.78),
}


# Generate points for each Gaussian and create a DataFrame
data_fs = []
for subject, mean in subjects_fs.items():
    points = rng.multivariate_normal(mean, covariance, points_per_gaussian)
    df = pd.DataFrame(points, columns=["x", "y"])
    df["label"] = subject
    data_fs.append(df)
# Combine all data into a single DataFrame
data_fs = pd.concat(data_fs)

position_fs = {
    "math": (-0.1, -0.1),
    "physics": (0.17, 0.05),
    "chemistry": (-0.2, -0.08),
    "philosopy": (0.2, -0.12),
    "history": (0.1, 0.1),
    "statistics": (0.2, 0.05),
    "machine\nlearning": (0.2, 0.1),
}


gs = GridSpec(1, 1)
ax = fig.add_subplot(gs[0])
sns.scatterplot(
    data=data_fs,
    x="x",
    y="y",
    hue="label",
    palette=custom_colors,
    s=30,
    alpha=dp_alpha,
)

points = rng.multivariate_normal((0.4, 0.5), [[1e-4, 0], [0, 1e-4]], 30)
sns.scatterplot(x=points[:, 0], y=points[:, 1], color="C3", s=30, alpha=dp_alpha)


plt.legend([], [], frameon=False)  # Remove the legend
plt.xticks([])  # Remove x-axis ticks
plt.yticks([])  # Remove y-axis ticks
plt.xlabel("")  # Remove x-axis label
plt.ylabel("")  # Remove y-axis label

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

# Add custom labels near each cluster
for label, mean in subjects_fs.items():
    pos = position_fs[label]
    plt.text(
        mean[0] + pos[0],
        mean[1] + pos[1],
        label,
        fontsize=9,
        color="black",
        weight="bold",
        va="center",
        ha="center",
    )
ax.set_title("few shot", weight="bold", fontsize=title_size)

# saddels
ax.plot(0.46, 0.32, marker="^", markersize=9, color="black")
ax.plot(0.48, 0.32, marker="^", markersize=9, color="black")


ax.plot(0.34, 0.55, marker="^", markersize=9, color="black")
ax.plot(0.31, 0.55, marker="^", markersize=9, color="black")

ax.plot(0.68, 0.3, marker="^", markersize=9, color="black")
ax.plot(0.65, 0.3, marker="^", markersize=9, color="black")

ax.plot(0.16, 0.7, marker="^", markersize=9, color="black")
ax.plot(0.19, 0.7, marker="^", markersize=9, color="black")

# hist-phil
ax.plot(0.76, 0.72, marker="^", markersize=9, color="black")
ax.plot(0.73, 0.72, marker="^", markersize=9, color="black")

ax.plot(0.6, 0.5, marker="^", markersize=9, color="black")
ax.plot(0.64, 0.5, marker="^", markersize=9, color="black")


# legend
ax.plot(0.06, 0.07, marker="^", markersize=7, color="black")
ax.plot(0.08, 0.07, marker="^", markersize=7, color="black")
plt.text(0.11, 0.06, "saddle point", fontsize=11, color="black", va="center")

rect = patches.Rectangle(
    (0, 0),
    1,
    1,
    transform=ax.transAxes,
    color=background_color,
    alpha=bg_alpha,
    zorder=0,
)
ax.add_patch(rect)


gs.tight_layout(fig, rect=[0.38, 0.0, 0.69, 1])


########################

# FINETUNED

########################

# Parameters for the Gaussians
std = 1.5e-3
covariance = [[std, 0], [0, std]]  # Same covariance matrix (variance)


subjects_ft = {
    "math": (0.45, 0.3),
    "physics": (0.55, 0.3),
    "chemistry": (0.55, 0.17),
    "philosopy": (0.75, 0.65),
    "history": (0.8, 0.75),
    "statistics": (0.25, 0.7),
    "machine\nlearning": (0.2, 0.8),
}

# Generate points for each Gaussian and create a DataFrame
data_ft = []
for subject, mean in subjects_ft.items():
    points = rng.multivariate_normal(mean, covariance, points_per_gaussian)
    df = pd.DataFrame(points, columns=["x", "y"])
    df["label"] = subject
    data_ft.append(df)
# Combine all data into a single DataFrame
data_ft = pd.concat(data_ft)

position_ft = {
    "math": (0.8, 0.2),
    "physics": (0.8, 0.15),
    "chemistry": (0.8, 0.1),
    "philosopy": (0.85, 0.95),
    "history": (0.85, 0.9),
    "statistics": (0.45, 0.8),
    "machine\nlearning": (0.45, 0.9),
}

gs = GridSpec(1, 1)
ax = fig.add_subplot(gs[0])
sns.scatterplot(
    data=data_ft,
    x="x",
    y="y",
    hue="label",
    palette=custom_colors,
    s=30,
    alpha=dp_alpha,
)

plt.legend([], [], frameon=False)  # Remove the legend
plt.xticks([])  # Remove x-axis ticks
plt.yticks([])  # Remove y-axis ticks
plt.xlabel("")  # Remove x-axis label
plt.ylabel("")  # Remove y-axis label

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)


# Add custom labels near each cluster
for label, mean in subjects_ft.items():
    pos = position_ft[label]
    plt.text(
        pos[0],
        pos[1],
        label,
        fontsize=9,
        color="black",
        weight="bold",
        va="center",
        ha="center",
    )
ax.set_title("finetuned", weight="bold", fontsize=title_size)


ax.plot(0.34, 0.5, marker="^", markersize=9, color="black")
ax.plot(0.3, 0.5, marker="^", markersize=9, color="black")

ax.plot(0.68, 0.5, marker="^", markersize=9, color="black")
ax.plot(0.64, 0.5, marker="^", markersize=9, color="black")


# legend
ax.plot(0.06, 0.07, marker="^", markersize=7, color="black")
ax.plot(0.08, 0.07, marker="^", markersize=7, color="black")
plt.text(0.11, 0.06, "saddle point", fontsize=11, color="black", va="center")


# ax.set_facecolor('#D2B48C')  # 'Tan' color
# Add a semi-transparent background color to the plot area
rect = patches.Rectangle(
    (0, 0),
    1,
    1,
    transform=ax.transAxes,
    color=background_color,
    alpha=bg_alpha,
    zorder=0,
)
ax.add_patch(rect)

gs.tight_layout(fig, rect=[0.69, 0, 1, 1])


plt.savefig(f"{path}/cartoon_7subjects.png", dpi=200)


#########################################################
#########################################################

# LETTERS

#########################################################
#########################################################
# Number of Gaussians and points per Gaussian
rng = np.random.default_rng(seed=42)
points_per_gaussian = 100
background_color = "peru"  # "antiquewhite"# "papayawhip"
bg_alpha = 0.1
dp_alpha = 1
title_size = 14
label_size = 14

# Define custom colors for each cluster
custom_colors = {
    "A": "C10",
    "B": "C8",
    "C": "C6",
    "D": "C9",
}

fig = plt.figure(figsize=(12, 3))

#########################################

# 0-SHOT

#########################################

# Parameters for the Gaussians
std = 4e-3
covariance = [[std, 0], [0, std]]  # Same covariance matrix (variance)


subjects = {
    "A": (0.6, 0.5),
    "B": (0.7, 0.5),
    "C": (0.6, 0.4),
    "D": (0.7, 0.3),
}


# Generate points for each Gaussian and create a DataFrame
data = []
for subject, mean in subjects.items():
    points = rng.multivariate_normal(mean, covariance, points_per_gaussian)
    df = pd.DataFrame(points, columns=["x", "y"])
    df["Answer"] = subject
    data.append(df)
# Combine all data into a single DataFrame
data = pd.concat(data)


gs = GridSpec(1, 1)
ax = fig.add_subplot(gs[0])


points = rng.multivariate_normal((0.25, 0.75), [[1.5e-3, 0], [0, 1.5e-3]], 40)
sns.scatterplot(x=points[:, 0], y=points[:, 1], color="C6", s=20, alpha=dp_alpha)
points = rng.multivariate_normal((0.25, 0.82), [[1.5e-3, 0], [0, 1.5e-3]], 40)
sns.scatterplot(x=points[:, 0], y=points[:, 1], color="C10", s=20, alpha=dp_alpha)


sns.scatterplot(
    data=data,
    x="x",
    y="y",
    hue="Answer",
    palette=custom_colors,
    s=30,
    alpha=dp_alpha,
)

# plt.legend([], [], frameon=False)  # Remove the legend

plt.xticks([])  # Remove x-axis ticks
plt.yticks([])  # Remove y-axis ticks
plt.xlabel("")  # Remove x-axis label
plt.ylabel("")  # Remove y-axis label

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_title("zero shot", weight="bold", fontsize=title_size)
ax.set_ylabel("late layers", weight="bold", fontsize=label_size)


# ax.set_facecolor('#D2B48C')  # 'Tan' color
# Add a semi-transparent background color to the plot area
rect = patches.Rectangle(
    (0, 0),
    1,
    1,
    transform=ax.transAxes,
    color=background_color,
    alpha=bg_alpha,
    zorder=0,
)
ax.add_patch(rect)

gs.tight_layout(fig, rect=[0.0, 0.0, 0.32, 1])


#########################

# FEW SHOT

#########################

# Parameters for the Gaussians
std = 2.2e-3
covariance = [[std, 0], [0, std]]  # Same covariance matrix (variance)


subjects_fs = {
    "A": (0.6, 0.7),
    "B": (0.75, 0.7),
    "C": (0.6, 0.55),
    "D": (0.75, 0.55),
}

# Generate points for each Gaussian and create a DataFrame
data_fs = []
for subject, mean in subjects_fs.items():
    points = rng.multivariate_normal(mean, covariance, points_per_gaussian)
    df = pd.DataFrame(points, columns=["x", "y"])
    df["Answer"] = subject
    data_fs.append(df)
# Combine all data into a single DataFrame
data_fs = pd.concat(data_fs)


gs = GridSpec(1, 1)
ax = fig.add_subplot(gs[0])


points = rng.multivariate_normal((0.25, 0.8), [[1e-3, 0], [0, 1e-3]], 50)
sns.scatterplot(x=points[:, 0], y=points[:, 1], color="C10", s=20, alpha=dp_alpha)
points = rng.multivariate_normal((0.25, 0.72), [[1e-3, 0], [0, 1e-3]], 50)
sns.scatterplot(x=points[:, 0], y=points[:, 1], color="C6", s=20, alpha=dp_alpha)


points = rng.multivariate_normal((0.6, 0.2), [[0.8e-3, 0], [0, 0.8e-3]], 40)
sns.scatterplot(x=points[:, 0], y=points[:, 1], color="C9", s=20, alpha=dp_alpha)


sns.scatterplot(
    data=data_fs,
    x="x",
    y="y",
    hue="Answer",
    palette=custom_colors,
    s=30,
    alpha=dp_alpha,
)


plt.xticks([])  # Remove x-axis ticks
plt.yticks([])  # Remove y-axis ticks
plt.xlabel("")  # Remove x-axis label
plt.ylabel("")  # Remove y-axis label

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_title("few shot", weight="bold", fontsize=title_size)

# ax.set_facecolor('#D2B48C')  # 'Tan' color
# Add a semi-transparent background color to the plot area
rect = patches.Rectangle(
    (0, 0),
    1,
    1,
    transform=ax.transAxes,
    color=background_color,
    alpha=bg_alpha,
    zorder=0,
)
ax.add_patch(rect)


gs.tight_layout(fig, rect=[0.38, 0.0, 0.69, 1])

#####################################àà

# FINETUNED

#######################################

# Parameters for the Gaussians
std = 1.8e-3
covariance = [[std, 0], [0, std]]  # Same covariance matrix (variance)


subjects_ft = {
    "A": (0.25, 0.7),
    "B": (0.75, 0.7),
    "C": (0.45, 0.35),
    "D": (0.8, 0.25),
}

# Generate points for each Gaussian and create a DataFrame
data_ft = []
for subject, mean in subjects_ft.items():
    points = rng.multivariate_normal(mean, covariance, points_per_gaussian)
    df = pd.DataFrame(points, columns=["x", "y"])
    df["Answer"] = subject
    data_ft.append(df)
# Combine all data into a single DataFrame
data_ft = pd.concat(data_ft)

gs = GridSpec(1, 1)
ax = fig.add_subplot(gs[0])
sns.scatterplot(
    data=data_ft,
    x="x",
    y="y",
    hue="Answer",
    palette=custom_colors,
    s=30,
    alpha=dp_alpha,
)
ax.legend(loc="lower left", title="Answer")
plt.xticks([])  # Remove x-axis ticks
plt.yticks([])  # Remove y-axis ticks
plt.xlabel("")  # Remove x-axis label
plt.ylabel("")  # Remove y-axis label

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_title("fine-tuned", weight="bold", fontsize=title_size)


# ax.set_facecolor('#D2B48C')  # 'Tan' color
# Add a semi-transparent background color to the plot area
rect = patches.Rectangle(
    (0, 0),
    1,
    1,
    transform=ax.transAxes,
    color=background_color,
    alpha=bg_alpha,
    zorder=0,
)
ax.add_patch(rect)


gs.tight_layout(fig, rect=[0.69, 0, 1, 1])

plt.savefig(f"{path}/cartoon_letters.png", dpi=200)
