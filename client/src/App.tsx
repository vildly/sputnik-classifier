import JSONForm from "./components/JSONForm"
import Loading, { LoadingColors } from "./components/Loading"

const ascii = [
    String.raw`.______.....______...__..__.....______...__...__.....__.....__..__....`,
    String.raw`/\..___\.../\..==.\./\.\/\.\.../\__.._\./\."-.\.\.../\.\.../\.\/./....`,
    String.raw`\.\___..\..\.\.._-/.\.\.\_\.\..\/_/\.\/.\.\.\-...\..\.\.\..\.\.._"-...`,
    String.raw`.\/\_____\..\.\_\....\.\_____\....\.\_\..\.\_\\"\_\..\.\_\..\.\_\.\_\.`,
    String.raw`..\/_____/...\/_/.....\/_____/.....\/_/...\/_/.\/_/...\/_/...\/_/\/_/.`,
]

function replaceBackground(text: string[]): string {
    // Replace every dot with a span element (using "class" not "className")
    return text
        .map(row => row.replaceAll(".", `<span class="text-black">.</span>`))
        .join("\n")
}

export default function App() {
    return (
        <div className="max-w-[80ch] h-screen mx-auto py-6 flex flex-col items-center">
            <pre
                className="w-min my-6 font-mono text-neutral-300 text-xs text-center whitespace-pre-wrap break-words"
                dangerouslySetInnerHTML={{ __html: replaceBackground(ascii) }}
            />
            <JSONForm />
            <Loading color={LoadingColors.BLUE} size={20} />
        </div>
    )
}
