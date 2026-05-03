import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", context="paper")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.family"] = "serif"

df_instancias = pd.read_csv("output/vns_instancias_todas_v1.csv")
df_ls1 = pd.read_csv("output/vns_estadisticas_ls1_comparadas_v1.csv")
df_grupo = pd.read_csv("output/vns_mejores_por_grupo_v1.csv")

palette = {
    "Pequeña": "#1f77b4",
    "Intermedia": "#d62728",
    "Completa": "#2ca02c"
}

# 1. Beneficio por instancia
plt.figure(figsize=(10, 5))
df_instancias_ordenado = df_instancias.sort_values("beneficio_musd", ascending=False)

ax = sns.barplot(
    data=df_instancias_ordenado,
    x="instancia",
    y="beneficio_musd",
    hue="grupo",
    palette=palette
)

plt.xticks(rotation=70)
plt.xlabel("Instancia")
plt.ylabel("Beneficio (M USD)")
plt.title("Beneficio obtenido por instancia con VNS")
plt.legend(title="Grupo")
plt.tight_layout()
plt.savefig("grafica_vns_beneficio_instancia.png", bbox_inches="tight")
plt.show()

# 2. Dispersión beneficio-distancia
plt.figure(figsize=(8, 5))
for grupo in ["Pequeña", "Intermedia"]:
    datos = df_instancias[df_instancias["grupo"] == grupo]
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
            fila["instancia"].replace("Instancia-", ""),
            (fila["distancia_km"], fila["beneficio_musd"]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=7
        )

plt.xlabel("Distancia (km)")
plt.ylabel("Beneficio (M USD)")
plt.title("Relación entre beneficio y distancia en VNS")
plt.legend(title="Grupo")
plt.tight_layout()
plt.savefig("grafica_vns_dispersion_beneficio_distancia.png", bbox_inches="tight")
plt.show()

# 3. Beneficio medio según ls1
plt.figure(figsize=(8, 5))
for grupo in ["Pequeña", "Intermedia"]:
    datos = df_ls1[df_ls1["grupo"] == grupo].sort_values("ls1")
    plt.plot(
        datos["ls1"],
        datos["beneficio_medio_musd"],
        marker="o",
        linewidth=2,
        label=grupo,
        color=palette[grupo]
    )

plt.xlabel("Valor de ls1")
plt.ylabel("Beneficio medio (M USD)")
plt.title("Beneficio medio según ls1")
plt.legend(title="Grupo")
plt.tight_layout()
plt.savefig("grafica_vns_beneficio_ls1.png", bbox_inches="tight")
plt.show()

# 4. Tiempo del mejor resultado por grupo
plt.figure(figsize=(7, 5))
orden_grupos = ["Pequeña", "Intermedia", "Completa"]
df_grupo["grupo"] = pd.Categorical(df_grupo["grupo"], categories=orden_grupos, ordered=True)
df_grupo = df_grupo.sort_values("grupo")

ax = sns.barplot(
    data=df_grupo,
    x="grupo",
    y="tiempo_s",
    hue="grupo",
    palette=palette,
    dodge=False,
    legend=False
)

for i, fila in enumerate(df_grupo.itertuples(index=False)):
    ax.text(
        i,
        fila.tiempo_s + 3,
        f"{fila.tiempo_s:.2f}",
        ha="center",
        fontsize=9
    )

plt.xlabel("Grupo de instancias")
plt.ylabel("Tiempo (s)")
plt.title("Tiempo del mejor resultado con VNS")
plt.tight_layout()
plt.savefig("grafica_vns_tiempo_grupo.png", bbox_inches="tight")
plt.show()