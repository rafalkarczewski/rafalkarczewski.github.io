document.addEventListener("DOMContentLoaded", function () {
  // Style footnotes
  document.querySelectorAll("d-footnote").forEach(function (footnote) {
    const shadowRoot = footnote.shadowRoot;

    // 1) Style the footnote reference number
    const footnoteNumber = shadowRoot.querySelector("sup > span");
    if (footnoteNumber) {
      footnoteNumber.setAttribute(
        "style",
        "color: var(--global-theme-color) !important; cursor: pointer;"
      );
    }

    // 2) Style the footnote container in the *footnote’s* shadow (NOT the hover box shadow)
    const footnoteContainer = shadowRoot.querySelector(".footnote-container");
    if (footnoteContainer) {
      // Inline styles approach
      footnoteContainer.style.backgroundColor = "var(--global-bg-color)";
      footnoteContainer.style.border = "1px solid var(--global-divider-color)";
      footnoteContainer.style.padding = "10px";
      footnoteContainer.style.borderRadius = "6px";
      footnoteContainer.style.color = "var(--global-text-color)";

      // If you want to force text color on all child elements, you can do:
      footnoteContainer.querySelectorAll("p, span, a").forEach(function (el) {
        el.style.color = "var(--global-text-color)";
      });
      // For link hover color, you cannot do it inline. 
      // You would either insert a rule into the footnote’s <style> or define a custom property.
    }

    // 3) If you still need to style the *d-hover-box* shadow itself, you can do so here:
    const hoverBox = shadowRoot.querySelector("d-hover-box");
    if (hoverBox) {
      const hoverBoxShadowStyle = hoverBox.shadowRoot.querySelector("style");
      if (hoverBoxShadowStyle && hoverBoxShadowStyle.sheet) {
        // Example: override the hover box panel if it has .panel or something similar
        // hoverBoxShadowStyle.sheet.insertRule(...);
      }
    }
  });

  // Style citations (this remains basically the same as you have now)
  document.querySelectorAll("d-cite").forEach(function (cite) {
    const shadowRoot = cite.shadowRoot;

    // Style the citation number
    const citationNumber = shadowRoot.querySelector(".citation-number");
    if (citationNumber) {
      citationNumber.setAttribute(
        "style",
        "color: var(--global-theme-color) !important;"
      );
    }

    // Style the d-cite hover box
    const hoverBox = shadowRoot.querySelector("d-hover-box");
    if (hoverBox) {
      const hoverBoxShadow = hoverBox.shadowRoot.querySelector("style").sheet;
      hoverBoxShadow.insertRule(
        ".panel { background-color: var(--global-bg-color) !important; border-color: var(--global-divider-color) !important; border-radius: 6px; }"
      );
      hoverBoxShadow.insertRule(
        ".panel p, .panel span { color: var(--global-text-color) !important; }"
      );
      hoverBoxShadow.insertRule(
        ".panel a { color: var(--global-text-color) !important; text-decoration: none; }"
      );
      hoverBoxShadow.insertRule(
        ".panel a:hover { color: var(--global-theme-color) !important; }"
      );
    }
  });
});
