import { HexColors } from "../lib/colors"

interface ASCIIArtProps {
    art: string[]
    bgPattern: RegExp
    bgColor: HexColors
}

/**
 * @param art When passing text with `\` (backslashes) it must be parsed using:
 *            `String.raw`\grave ...text... \grave
 */
export default function ASCIIArt({ art, bgPattern, bgColor }: ASCIIArtProps) {
    function replaceBackground(text: string, pattern: RegExp, color: HexColors): string {
        // $& in the replace method refers to the matched substring
        // (the part of the text that matches the pattern).
        // This ensures that only the matched characters are wrapped with the <span>.
        //
        // You can also inject the dynamically matched character by doing it like this:
        // .replace(pattern, match => `<span ... >${match}</span>`)
        return text.replace(pattern, `<span style="color: ${color}">$&</span>`)
    }

    const htmlContent = art.map(row => replaceBackground(row, bgPattern, bgColor)).join("\n")
    return (
        <pre
            className="w-min my-6 font-mono text-neutral-300 text-xs text-center whitespace-pre-wrap break-words"
            dangerouslySetInnerHTML={{ __html: htmlContent }}
        />
    )
}
