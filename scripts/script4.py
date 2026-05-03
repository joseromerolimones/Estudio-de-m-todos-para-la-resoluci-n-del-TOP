import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", context="paper")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.family"] = "serif"

df_instancias = pd.read_csv("output/exacto_instancias_todas_v1.csv")
df_medias = pd.read_csv("output/exacto_medias_por_tamano_v1.csv")
df_mejores = pd.read_csv("output/exacto_mejores_por_grupo_v1.csv")

palette = {
    "Pequeña": "#1f77b4",
    "Mediana": "#d62728",
    "Completa": "#2ca02c"
}

orden = ["Pequeña", "Mediana", "Completa"]

# 1. Beneficio por instancia
plt.figure(figsize=(10, 5))
df_benef = df_instancias[df_instancias["tamano"] != "Completa"].sort_values("beneficio_musd", ascending=False)

sns.barplot(
    data=df_benef,
    x="instancia",
    y="beneficio_musd",
    hue="tamano",
    palette=palette
)

plt.xticks(rotation=70)
plt.xlabel("Instancia")
plt.ylabel("Beneficio (M USD)")
plt.title("Beneficio obtenido por instancia con Pyomo")
plt.legend(title="Tamaño")
plt.tight_layout()
plt.savefig("grafica_exacto_beneficio_instancia.png", bbox_inches="tight")
plt.show()
plt.close()

# 2. Dispersión beneficio-distancia
plt.figure(figsize=(8, 5))
for tamano in ["Pequeña", "Mediana"]:
    datos = df_instancias[df_instancias["tamano"] == tamano]
    plt.scatter(
        datos["distancia_km"],
        datos["beneficio_musd"],
        s=70,
        label=tamano,
        color=palette[tamano],
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
plt.title("Relación entre beneficio y distancia con Pyomo")
plt.legend(title="Tamaño")
plt.tight_layout()
plt.savefig("grafica_exacto_dispersion_beneficio_distancia.png", bbox_inches="tight")
plt.show()
plt.close()

# 3. Beneficio medio por tamaño
plt.figure(figsize=(7, 5))
df_medias["tamano"] = pd.Categorical(df_medias["tamano"], categories=orden, ordered=True)
df_medias = df_medias.sort_values("tamano")

ax = sns.barplot(
    data=df_medias,
    x="tamano",
    y="beneficio_medio_musd",
    hue="tamano",
    palette=palette,
    dodge=False,
    legend=False
)

for i, fila in enumerate(df_medias.itertuples(index=False)):
    ax.text(
        i,
        fila.beneficio_medio_musd + 10,
        f"{fila.beneficio_medio_musd:.2f}",
        ha="center",
        fontsize=9
    )

plt.xlabel("Tamaño de instancia")
plt.ylabel("Beneficio medio (M USD)")
plt.title("Beneficio medio por tamaño con Pyomo")
plt.tight_layout()
plt.savefig("grafica_exacto_beneficio_medio_tamano.png", bbox_inches="tight")
plt.show()
plt.close()

# 4. Tiempo del mejor resultado por tamaño
plt.figure(figsize=(7, 5))
df_mejores["tamano"] = pd.Categorical(df_mejores["tamano"], categories=orden, ordered=True)
df_mejores = df_mejores.sort_values("tamano")

ax = sns.barplot(
    data=df_mejores,
    x="tamano",
    y="tiempo_s",
    hue="tamano",
    palette=palette,
    dodge=False,
    legend=False
)

for i, fila in enumerate(df_mejores.itertuples(index=False)):
    etiqueta = f"{fila.tiempo_s:.4f}" if fila.tiempo_s < 1 else f"{fila.tiempo_s:.2f}"
    ax.text(
        i,
        fila.tiempo_s + 0.03,
        etiqueta,
        ha="center",
        fontsize=9
    )

plt.xlabel("Tamaño de instancia")
plt.ylabel("Tiempo (s)")
plt.title("Tiempo del mejor resultado con Pyomo")
plt.tight_layout()
plt.savefig("grafica_exacto_tiempo_mejor_tamano.png", bbox_inches="tight")
plt.show()
plt.close()