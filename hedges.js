import { hedges } from 'hedges'
import fs from 'fs'
import path from 'path'

// save this to a file in the path of /configs/hedges.json
const outputPath = path.join(process.cwd(), 'configs', 'hedges.json')
const hedgesJson = JSON.stringify(hedges, null, 4)
fs.writeFileSync(outputPath, hedgesJson, 'utf8')

console.log(`Hedges saved to ${outputPath}`)
