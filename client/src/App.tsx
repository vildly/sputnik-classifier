import JSONForm from "./components/JSONForm"

 export default function App() {
    return (
        <div className="max-w-[80ch] h-screen mx-auto py-6 flex flex-col items-center">
            <pre className="w-min my-6 font-mono text-neutral-500 text-xs text-center whitespace-pre-wrap break-words">
                .______.....______...__..__.....______...__...__.....__.....__..__....
                /\..___\.../\..==.\./\.\/\.\.../\__.._\./\."-.\.\.../\.\.../\.\/./....
                \.\___..\..\.\.._-/.\.\.\_\.\..\/_/\.\/.\.\.\-...\..\.\.\..\.\.._"-...
                .\/\_____\..\.\_\....\.\_____\....\.\_\..\.\_\\"\_\..\.\_\..\.\_\.\_\.
                ..\/_____/...\/_/.....\/_____/.....\/_/...\/_/.\/_/...\/_/...\/_/\/_/.
            </pre>
            <JSONForm />
        </div>
    )
}
