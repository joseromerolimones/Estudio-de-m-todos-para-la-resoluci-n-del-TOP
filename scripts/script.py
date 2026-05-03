import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", context="paper")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.family"] = "serif"

df_alpha = pd.read_csv("output/ahorro_mejores_por_alpha_todos_v2.csv")
df_medias = pd.read_csv("output/ahorro_sensibilidad_medias_v2.csv")
df_grupo = pd.read_csv("output/ahorro_mejores_por_grupo_v2.csv")

palette = {
    "Pequeña": "#1f77b4",
    "Pequeñas": "#1f77b4",
    "Intermedia": "#d62728",
    "Intermedias": "#d62728",
    "Completa": "#2ca02c"
}

orden_grupos = ["Pequeña", "Intermedia", "Completa"]
orden_grupos_barras = ["Pequeñas", "Intermedias", "Completa"]

# 1. Beneficio vs alpha
plt.figure(figsize=(8, 5))
for grupo in ["Pequeña", "Intermedia", "Completa"]:
    datos = df_alpha[df_alpha["grupo"] == grupo].sort_values("alpha")
    plt.plot(
        datos["alpha"],
        datos["beneficio_musd"],
        marker="o",
        linewidth=2,
        label=grupo,
        color=palette[grupo]
    )

plt.xlabel("Valor de α")
plt.ylabel("Beneficio (M USD)")
plt.title("Beneficio obtenido según α")
plt.legend(title="Grupo")
plt.tight_layout()
plt.savefig("grafica_beneficio_alpha.png", bbox_inches="tight")
plt.show()

# 2. Distancia vs alpha
plt.figure(figsize=(8, 5))
for grupo in ["Pequeña", "Intermedia", "Completa"]:
    datos = df_alpha[df_alpha["grupo"] == grupo].sort_values("alpha")
    plt.plot(
        datos["alpha"],
        datos["distancia_km"],
        marker="o",
        linewidth=2,
        label=grupo,
        color=palette[grupo]
    )

plt.xlabel("Valor de α")
plt.ylabel("Distancia (km)")
plt.title("Distancia recorrida según α")
plt.legend(title="Grupo")
plt.tight_layout()
plt.savefig("grafica_distancia_alpha.png", bbox_inches="tight")
plt.show()

# 3. Dispersión beneficio-distancia
plt.figure(figsize=(8, 5))
for grupo in ["Pequeña", "Intermedia", "Completa"]:
    datos = df_alpha[df_alpha["grupo"] == grupo]
    plt.scatter(
        datos["distancia_km"],
        datos["beneficio_musd"],
        s=70,
        label=grupo,
        color=palette[grupo],
        alpha=0.85
    )
    for _, fila in datos.iterrows():
        plt.annotate(
            f"{fila['alpha']:.1f}",
            (fila["distancia_km"], fila["beneficio_musd"]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8
        )

plt.xlabel("Distancia (km)")
plt.ylabel("Beneficio (M USD)")
plt.title("Relación entre beneficio y distancia")
plt.legend(title="Grupo")
plt.tight_layout()
plt.savefig("grafica_dispersion_beneficio_distancia.png", bbox_inches="tight")
plt.show()

# 4. Mejor tiempo por grupo
plt.figure(figsize=(7, 5))
df_grupo_ordenado = df_grupo.copy()
df_grupo_ordenado["grupo"] = pd.Categorical(
    df_grupo_ordenado["grupo"],
    categories=orden_grupos_barras,
    ordered=True
)
df_grupo_ordenado = df_grupo_ordenado.sort_values("grupo")

ax = sns.barplot(
    data=df_grupo_ordenado,
    x="grupo",
    y="tiempo_s",
    hue="grupo",
    palette=palette,
    dodge=False,
    legend=False
)

for i, fila in df_grupo_ordenado.iterrows():
    ax.text(
        x=list(df_grupo_ordenado.index).index(i),
        y=fila["tiempo_s"] + 0.03,
        s=f"{fila['tiempo_s']:.2f}",
        ha="center",
        fontsize=9
    )

plt.xlabel("Grupo de instancias")
plt.ylabel("Tiempo (s)")
plt.title("Tiempo de ejecución del mejor resultado")
plt.tight_layout()
plt.savefig("grafica_tiempo_grupo.png", bbox_inches="tight")
plt.show()