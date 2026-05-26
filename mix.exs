defmodule MlMusings.MixProject do
  use Mix.Project

  def project do
    [
      app: :ml_musings,
      version: "0.1.0",
      elixir: "~> 1.14",
      start_permanent: Mix.env() == :prod,
      description: description(),
      package: package(),
      deps: deps()
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger]
    ]
  end

  defp description do
    """
    TRINITY Foundations: A hands-on curriculum exploring high-dimensional geometry,
    self-attention routing, PEFT (LoRA and SVD surgery), evolution strategies, and sparse Mixture of Experts (MoE) gating.
    """
  end

  defp package do
    [
      name: "ml_musings",
      files: ~w(
        mix.exs
        README.md
        LICENSE
        CHANGELOG.md
        assets
        01_list_math.exs
        02_tensor_math.exs
        03_compiler.exs
        04_dot_product.exs
        05_linear_probe.exs
        06_self_attention.exs
        07_softmax_collapse.exs
        08_lora_and_svd.exs
        09_non_redundant_compression.exs
        10_evolution_strategy.exs
        11_rosenbrock_es.exs
        12_moe_gating.exs
        13_loss_curve.exs
        14_mini_trinity.exs
        quasi_orthogonality.exs

        hoeffding_bound.exs
      ),
      licenses: ["MIT"],
      links: %{
        "GitHub" => "https://github.com/nshkrdotcom/ml_musings"
      }
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:nx, "~> 0.12.0"},
      {:exla, "~> 0.12.0"},
      {:ex_doc, "~> 0.31", only: :dev, runtime: false}
    ]
  end
end
