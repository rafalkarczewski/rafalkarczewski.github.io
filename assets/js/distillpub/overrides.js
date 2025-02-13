document.addEventListener("DOMContentLoaded", function () {
  // Style footnotes
  document.querySelectorAll("d-footnote").forEach(function (footnote) {
    const shadowRoot = footnote.shadowRoot;

    // Style the footnote reference number
    shadowRoot.querySelector("sup > span").setAttribute(
      "style",
      "color: var(--global-theme-color) !important; cursor: pointer;"
    );

    // Style the hover box
    const hoverBoxStyle = shadowRoot.querySelector("d-hover-box").shadowRoot.querySelector("style").sheet;
    hoverBoxStyle.insertRule(
      ".footnote-container {background-color: var(--global-bg-color) !important; border: 1px solid var(--global-divider-color) !important; padding: 10px; border-radius: 6px;}"
    );
    hoverBoxStyle.insertRule(
      ".footnote-container p, .footnote-container span {color: var(--global-text-color) !important;}"
    );
    hoverBoxStyle.insertRule(
      ".footnote-container a {color: var(--global-text-color) !important; text-decoration: none;}"
    );
    hoverBoxStyle.insertRule(
      ".footnote-container a:hover {color: var(--global-theme-color) !important;}"
    );
  });

  // Style citations
  document.querySelectorAll("d-cite").forEach(function (cite) {
    const shadowRoot = cite.shadowRoot;

    // Style the citation number
    shadowRoot.querySelector(".citation-number").setAttribute(
      "style",
      "color: var(--global-theme-color) !important;"
    );

    // Style the hover box
    const hoverBoxStyle = shadowRoot.querySelector("d-hover-box").shadowRoot.querySelector("style").sheet;
    hoverBoxStyle.insertRule(
      ".panel {background-color: var(--global-bg-color) !important; border-color: var(--global-divider-color) !important;}"
    );
    hoverBoxStyle.insertRule(
      ".panel p, .panel span {color: var(--global-text-color) !important;}"
    );
    hoverBoxStyle.insertRule(
      ".panel a {color: var(--global-text-color) !important; text-decoration: none;}"
    );
    hoverBoxStyle.insertRule(
      ".panel a:hover {color: var(--global-theme-color) !important;}"
    );
  });
});
