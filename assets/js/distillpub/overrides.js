document.addEventListener("DOMContentLoaded", function () {
  document.querySelectorAll("d-cite").forEach(function (cite) {
    const shadowRoot = cite.shadowRoot;

    // Override the citation number color
    shadowRoot.querySelector(".citation-number").setAttribute(
      "style",
      "color: var(--global-theme-color) !important;"
    );

    // Override the hover box styles
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

    // Override list items inside the hover box
    hoverBoxStyle.insertRule(
      "ul li {padding: 10px 8px; border-bottom: 1px solid var(--global-divider-color) !important;}"
    );
    hoverBoxStyle.insertRule(
      "ul li:last-of-type {border-bottom: none;}"
    );
  });
});
