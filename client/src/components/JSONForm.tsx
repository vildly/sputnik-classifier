import { useState } from "react"
import JSONInput from "./JSONInput"

export default function JSONForm() {
    const [value, setValue] = useState<string>("")

    function handleSubmit(e: React.FormEvent<HTMLFormElement>): void {
        e.preventDefault()
        console.log(`Submitted value: ${value}`)
        setValue("")
    }

    return (
        <div className="w-full h-full">
            <form onSubmit={handleSubmit} className="h-full flex flex-col space-y-2">
                <JSONInput
                    setValue={setValue}
                    value={value}
                    label="JSON"
                    placeholder="Enter JSON here..."
                />
                <button
                    type="submit"
                    className="w-fit px-4 py-2 text-white bg-blue-500 rounded hover:bg-blue-600 transition"
                >
                    Submit
                </button>
            </form>
        </div>
    )
}
