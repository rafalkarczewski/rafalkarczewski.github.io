$(document).ready(function () {
  // Override styles of the footnotes.
  document.querySelectorAll("d-footnote").forEach(function (footnote) {
    // Style the footnote number
    footnote.shadowRoot.querySelector("sup > span").setAttribute("style", "color: var(--global-theme-color);");

    // Style the hover box (background and border)
    footnote.shadowRoot.querySelector("d-hover-box").shadowRoot.querySelector("style").sheet
      .insertRule(".panel {background-color: var(--global-bg-color) !important; border-color: var(--global-divider-color) !important;}");
    
    // Style text inside the hover box
    footnote.shadowRoot.querySelector("d-hover-box").shadowRoot.querySelector("style").sheet
      .insertRule(".panel p, .panel span {color: var(--global-text-color) !important;}");
  });

  // Override styles of the citations.
  document.querySelectorAll("d-cite").forEach(function (cite) {
    // Style the citation number ([1])
    cite.shadowRoot.querySelector("div > span").setAttribute("style", "color: var(--global-theme-color);");

    // Style the hover box (background and border)
    cite.shadowRoot.querySelector("d-hover-box").shadowRoot.querySelector("style").sheet
      .insertRule(".panel {background-color: var(--global-bg-color) !important; border-color: var(--global-divider-color) !important;}");

    // Style text inside the hover box
    cite.shadowRoot.querySelector("d-hover-box").shadowRoot.querySelector("style").sheet
      .insertRule(".panel p, .panel span {color: var(--global-text-color) !important;}");

    // Style links inside the hover box
    cite.shadowRoot.querySelector("d-hover-box").shadowRoot.querySelector("style").sheet
      .insertRule(".panel a {color: var(--global-text-color) !important; text-decoration: none;}");
    cite.shadowRoot.querySelector("d-hover-box").shadowRoot.querySelector("style").sheet
      .insertRule(".panel a:hover {color: var(--global-theme-color) !important;}");
  });
});
