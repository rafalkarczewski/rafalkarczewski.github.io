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
    const hoverBox = shadowRoot.querySelector("d-hover-box");
    if (hoverBox) {
      const hoverBoxShadow = hoverBox.shadowRoot.querySelector("style").sheet;

      // Ensure the footnote container has the correct styles
      hoverBoxShadow.insertRule(
        ".footnote-container {background-color: var(--global-bg-color) !important; border: 1px solid var(--global-divider-color) !important; padding: 10px; border-radius: 6px;}"
      );

      // Ensure text within the container uses the correct styles
      hoverBoxShadow.insertRule(
        ".footnote-container p, .footnote-container span {color: var(--global-text-color) !important;}"
      );

      // Ensure links within the container are styled correctly
      hoverBoxShadow.insertRule(
        ".footnote-container a {color: var(--global-text-color) !important; text-decoration: none;}"
      );
      hoverBoxShadow.insertRule(
        ".footnote-container a:hover {color: var(--global-theme-color) !important;}"
      );
    }
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
    const hoverBox = shadowRoot.querySelector("d-hover-box");
    if (hoverBox) {
      const hoverBoxShadow = hoverBox.shadowRoot.querySelector("style").sheet;

      hoverBoxShadow.insertRule(
        ".panel {background-color: var(--global-bg-color) !important; border-color: var(--global-divider-color) !important; border-radius: 6px;}"
      );
      hoverBoxShadow.insertRule(
        ".panel p, .panel span {color: var(--global-text-color) !important;}"
      );
      hoverBoxShadow.insertRule(
        ".panel a {color: var(--global-text-color) !important; text-decoration: none;}"
      );
      hoverBoxShadow.insertRule(
        ".panel a:hover {color: var(--global-theme-color) !important;}"
      );
    }
  });
});
