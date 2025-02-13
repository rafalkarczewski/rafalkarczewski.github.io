/**
 * overrides.js
 *
 * Ensure the color/background for <d-footnote> and <d-cite> (including
 * their <d-hover-box> popups) are forced to custom global variables.
 */
(function () {
  // --- Main entry point ----------------------------------
  document.addEventListener("DOMContentLoaded", () => {
    // Style any existing footnotes/citations
    styleAllFootnotes();
    styleAllCitations();

    // Watch for newly added or re-inserted nodes
    const observer = new MutationObserver(() => {
      styleAllFootnotes();
      styleAllCitations();
    });
    observer.observe(document.body, { childList: true, subtree: true });
  });

  // --- Helpers -------------------------------------------
  function styleAllFootnotes() {
    document.querySelectorAll("d-footnote").forEach((footnote) => {
      const root = footnote.shadowRoot;
      if (!root) return;

      // 1) Style the footnote reference number (<sup><span>…</span></sup>)
      const footnoteNumber = root.querySelector("sup > span");
      if (footnoteNumber) {
        footnoteNumber.style.setProperty("color", "var(--global-theme-color)", "important");
        footnoteNumber.style.setProperty("cursor", "pointer");
      }

      // 2) Style the footnote container in the footnote's own shadow
      const footnoteContainer = root.querySelector(".footnote-container");
      if (footnoteContainer) {
        footnoteContainer.style.setProperty("background-color", "var(--global-bg-color)", "important");
        footnoteContainer.style.setProperty("border", "1px solid var(--global-divider-color)", "important");
        footnoteContainer.style.setProperty("padding", "10px", "important");
        footnoteContainer.style.setProperty("border-radius", "6px", "important");
        footnoteContainer.style.setProperty("color", "var(--global-text-color)", "important");
      }

      // 3) Style the footnote's <d-hover-box> shadow (.panel)
      const hoverBox = root.querySelector("d-hover-box");
      if (hoverBox && hoverBox.shadowRoot) {
        const styleEl = hoverBox.shadowRoot.querySelector("style");
        if (styleEl && styleEl.sheet) {
          try {
            styleEl.sheet.insertRule(
              ".panel { background-color: var(--global-bg-color) !important; border-color: var(--global-divider-color) !important; border-radius: 6px; }",
              styleEl.sheet.cssRules.length
            );
            styleEl.sheet.insertRule(
              ".panel p, .panel span { color: var(--global-text-color) !important; }",
              styleEl.sheet.cssRules.length
            );
            styleEl.sheet.insertRule(
              ".panel a { color: var(--global-text-color) !important; text-decoration: none; }",
              styleEl.sheet.cssRules.length
            );
            styleEl.sheet.insertRule(
              ".panel a:hover { color: var(--global-theme-color) !important; }",
              styleEl.sheet.cssRules.length
            );
          } catch (err) {
            console.warn("Could not insert footnote hoverBox rules:", err);
          }
        }
      }
    });
  }

  function styleAllCitations() {
    document.querySelectorAll("d-cite").forEach((cite) => {
      const root = cite.shadowRoot;
      if (!root) return;

      // 1) Style the citation number
      const citationNumber = root.querySelector(".citation-number");
      if (citationNumber) {
        citationNumber.style.setProperty("color", "var(--global-theme-color)", "important");
        citationNumber.style.setProperty("cursor", "pointer");
      }

      // 2) Style the citation's <d-hover-box> shadow (.panel)
      const hoverBox = root.querySelector("d-hover-box");
      if (hoverBox && hoverBox.shadowRoot) {
        const styleEl = hoverBox.shadowRoot.querySelector("style");
        if (styleEl && styleEl.sheet) {
          try {
            styleEl.sheet.insertRule(
              ".panel { background-color: var(--global-bg-color) !important; border-color: var(--global-divider-color) !important; border-radius: 6px; }",
              styleEl.sheet.cssRules.length
            );
            styleEl.sheet.insertRule(
              ".panel p, .panel span { color: var(--global-text-color) !important; }",
              styleEl.sheet.cssRules.length
            );
            styleEl.sheet.insertRule(
              ".panel a { color: var(--global-text-color) !important; text-decoration: none; }",
              styleEl.sheet.cssRules.length
            );
            styleEl.sheet.insertRule(
              ".panel a:hover { color: var(--global-theme-color) !important; }",
              styleEl.sheet.cssRules.length
            );
          } catch (err) {
            console.warn("Could not insert citation hoverBox rules:", err);
          }
        }
      }
    });
  }
})();
